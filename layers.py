# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


version_str = torch.__version__
major_version = version_str.split('.')[0]
def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    # Get rotation matrix R from axis-angle representation
    R = rot_from_axisangle(axisangle)
    # Clone translation vector
    t = translation.clone()
    # If invert is True, compute inverse transform (when frame_id < 0)

    if invert:
        # Inverse of rotation matrix equals its transpose
        R = R.transpose(1, 2)
        # Negate translation vector
        t *= -1
    # Get translation matrix T from translation vector t
    T = get_translation_matrix(t)
    #print(R.max(),R.min(),T.max(),T.min().dtype)
    #print("axisangle, translation, invert", axisangle.dtype, translation.dtype, invert)
    # torch.set_float32_matmul_precision("highest")

    if major_version == '1':


        if invert:
            # If invert is True, apply rotation first then translation (inverse transform)
            M = torch.matmul(R, T)
        else:
            # Otherwise, apply translation first then rotation (forward transform)
            M = torch.matmul(T, R)

    else:
        with torch.autocast(device_type="cuda",dtype=torch.float32):

            if invert:
                # If invert is True, apply rotation first then translation (inverse transform)
                M = torch.matmul(R, T)
            else:
                # Otherwise, apply translation first then rotation (forward transform)
                M = torch.matmul(T, R)

        #print( M.max(), M.min())
        #print(M.dtype)
    # Return the final 4x4 transformation matrix M
    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        """
        Initialize the BackprojectDepth layer.

        Args:
            batch_size (int): Batch size.
            height (int): Height of the depth image.
            width (int): Width of the depth image.
        """
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        # Create pixel coordinate grid (u, v).
        # np.meshgrid returns two arrays representing column indices and row indices.
        # indexing='xy' ensures the first array corresponds to the x-axis (width), the second to the y-axis (height).
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        # Stack the grid into an array of shape (2, height, width), where the first dimension contains u coordinates and the second contains v coordinates.
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        # Convert numpy array to PyTorch tensor and register it as a module parameter.
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)
        # Create a tensor of ones with shape (batch_size, 1, height * width) for homogeneous coordinate representation.
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)
        # Extract u and v coordinates from id_coords and flatten them into 1D arrays.
        # self.id_coords[0] is the u coordinate, self.id_coords[1] is the v coordinate.
        # .view(-1) flattens them into 1D arrays.
        # torch.stack stacks u and v coordinates together, forming a tensor of shape (2, height * width).
        # torch.unsqueeze(..., 0) adds a dimension at the outermost level, making the shape (1, 2, height * width).
        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        # Repeat pixel coordinates along the batch dimension batch_size times, making the shape (batch_size, 2, height * width).
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        # Concatenate pixel coordinates and ones tensor along dimension 1, forming homogeneous pixel coordinates (u, v, 1) with shape (batch_size, 3, height * width).
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        """
        Perform backprojection from depth image to point cloud.

        Args:
            depth (torch.Tensor): Depth image tensor of shape (batch_size, 1, height, width).
            inv_K (torch.Tensor): Inverse camera intrinsic matrix tensor of shape (batch_size, 3, 3).

        Returns:
            torch.Tensor: Point cloud tensor in camera coordinates of shape (batch_size, 4, height * width),
                          where each point is represented as homogeneous coordinates (X, Y, Z, 1).
        """
        # Multiply the top 3x3 part of the inverse camera intrinsic matrix with homogeneous pixel coordinates.
        # This transforms pixel coordinates from the image plane to rays in the camera coordinate system (without considering depth yet).
        # inv_K[:, :3, :3] has shape (batch_size, 3, 3), self.pix_coords has shape (batch_size, 3, height * width),
        # the product gives cam_points of shape (batch_size, 3, height * width).
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        # Flatten the depth image into a tensor of shape (batch_size, 1, height * width),
        # and multiply it element-wise with cam_points to get the actual 3D coordinates (X, Y, Z) of each point in the camera coordinate system.
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        # Append a row of ones to the camera coordinate point cloud, forming homogeneous coordinates (X, Y, Z, 1),
        # for subsequent transformation operations.
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        """
        Perform 3D point projection.

        Args:
            points (torch.Tensor): 3D point tensor of shape (batch_size, 4, N), where N is the number of points.
                                   The last dimension of each point is assumed to be homogeneous coordinates (x, y, z, 1).
            K (torch.Tensor): Camera intrinsic matrix tensor of shape (batch_size, 3, 3).
            T (torch.Tensor): Camera extrinsic transformation matrix tensor of shape (batch_size, 4, 4).
                                   It transforms points from the world coordinate system to the camera coordinate system.

        Returns:
            torch.Tensor: Normalized pixel coordinate tensor of shape (batch_size, height, width, 2),
                          with values in the range [-1, 1].
        """
        # Compute the projection matrix P. The projection matrix maps points from the world coordinate system
        # directly to the camera's normalized image plane.
        # P = K * [R|t], where K is the intrinsic matrix and [R|t] is the rotation matrix and translation vector from the extrinsic matrix T.
        # Here we directly multiply K by the first three columns of T (corresponding to rotation and translation).

        P = torch.matmul(K, T)[:, :3, :] ## K has shape (batch_size, 3, 3), T has shape (batch_size, 4, 4),
                                         # after multiplication take the first 3 rows and all columns, resulting in projection matrix P of shape (batch_size, 3, 4).
        # Transform 3D points from the world coordinate system to the camera coordinate system.
        # This is done by multiplying the projection matrix P with the 3D points.
        # points has shape (batch_size, 4, N), P has shape (batch_size, 3, 4),
        # the product gives points in the camera coordinate system of shape (batch_size, 3, N).
        cam_points = torch.matmul(P, points)

        # # Project points from the camera coordinate system onto the normalized image plane.
        #         # This is done by dividing x and y coordinates by the z coordinate.
        #         # cam_points[:, :2, :] selects x and y coordinates, cam_points[:, 2, :].unsqueeze(1) selects z coordinate and adds a dimension for broadcast division.
        #         # Adding self.eps prevents division by zero.
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        # # At this point pix_coords has shape (batch_size, 2, N), where N is the number of points.
        # We assume these points correspond to pixels in the image, so reshape to (batch_size, 2, height, width).
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        # Permute dimensions from (batch_size, 2, height, width) to (batch_size, height, width, 2),
        # so that the last dimension contains x and y coordinates.
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        # Normalize pixel coordinates to the range [-1, 1].
        # Original pixel coordinate ranges are [0, width-1] and [0, height-1].
        # First normalize x coordinates to [0, 1] range.

        pix_coords[..., 0] /= self.width - 1
        # Then normalize y coordinates to [0, 1] range.
        pix_coords[..., 1] /= self.height - 1
        # Finally, adjust the range from [0, 1] to [-1, 1].
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords




def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

def upsample(x,scale_factor=2,mode="nearest"):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)


def get_smooth_loss(disp, img,mask=0):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    if not isinstance(mask,int):
        mask = torch.mean(torch.abs((mask[:, :, :-1, :] + mask[:, :, 1:, :]) / 2), 1, keepdim=True)
        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= (100 * (1 - mask) + mask) * torch.exp(-grad_img_y)#torch.exp(-grad_img_y)
    else:

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)


    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        """Layer to compute the SSIM loss between a pair of images
            """
        super(SSIM, self).__init__()
        # Average pooling layers for computing means, kernel size 3x3, stride 1
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)# For computing local covariance of x and y

        # Reflection padding, adding 1 pixel of padding at image edges to avoid edge effects
        self.refl = nn.ReflectionPad2d(1)

        # Constant terms for numerical stability, preventing division by zero
        self.C1 = 0.01 ** 2 # Constant term in luminance comparison
        self.C2 = 0.03 ** 2 # Constant term in contrast comparison

    def forward(self, x, y):
        # Apply reflection padding to input images
        x = self.refl(x)
        y = self.refl(y)

        # Compute local means
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        # Compute local variances and covariance (contrast and structure)
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        # Compute numerator and denominator of the SSIM formula
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        # Compute SSIM loss, clamp values to [0, 1] range
        # Note: returns (1-SSIM)/2, converting similarity to a loss value
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class GMSD(nn.Module):
    """Layer to compute Gradient Magnitude Similarity Deviation (GMSD) between a pair of images
    GMSD focuses on gradient information and is particularly sensitive to structural changes
    """

    def __init__(self):
        super(GMSD, self).__init__()
        # Define Sobel operators for computing gradients
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)

        # Reflection padding, adding 1 pixel of padding at image edges to avoid edge effects
        self.refl = nn.ReflectionPad2d(1)

        # Constant term for numerical stability
        self.eps = 1e-6

    def forward(self, x, y):
        # Ensure input is a 4D tensor [batch, channels, height, width]
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if len(y.shape) == 3:
            y = y.unsqueeze(0)

        # If color image, convert to grayscale
        if x.shape[1] == 3:
            # RGB to grayscale conversion weights
            weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
            x_gray = (x * weights).sum(dim=1, keepdim=True)
            y_gray = (y * weights).sum(dim=1, keepdim=True)
        else:
            x_gray = x
            y_gray = y

        # Apply reflection padding to input images
        x_gray = self.refl(x_gray)
        y_gray = self.refl(y_gray)

        # Move Sobel operators to the same device as input
        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)

        # Compute x-direction gradients
        grad_x_x = F.conv2d(x_gray, sobel_x, padding=0)
        grad_x_y = F.conv2d(y_gray, sobel_x, padding=0)

        # Compute y-direction gradients
        grad_y_x = F.conv2d(x_gray, sobel_y, padding=0)
        grad_y_y = F.conv2d(y_gray, sobel_y, padding=0)

        # Compute gradient magnitudes
        grad_mag_x = torch.sqrt(grad_x_x ** 2 + grad_y_x ** 2 + self.eps)
        grad_mag_y = torch.sqrt(grad_x_y ** 2 + grad_y_y ** 2 + self.eps)

        # Compute gradient magnitude similarity
        quality_map = (2 * grad_mag_x * grad_mag_y + self.eps) / (grad_mag_x ** 2 + grad_mag_y ** 2 + self.eps)

        # Compute GMSD at each position (using local standard deviation instead of global)
        # Maintain the same spatial dimensions as input
        batch_size = quality_map.shape[0]
        channels = quality_map.shape[1]
        height = quality_map.shape[2]
        width = quality_map.shape[3]

        # Compute global GMSD value
        global_gmsd = torch.std(quality_map.view(batch_size, -1), dim=1)

        # Create output tensor with the same shape as quality_map, filled with global GMSD value
        gmsd_map = global_gmsd.view(batch_size, 1, 1, 1).expand(batch_size, channels, height, width)

        # Convert GMSD to loss value, range [0, 1]
        return torch.clamp(gmsd_map, 0, 1)
import math
class FSIM(nn.Module):
    """Layer to compute Feature Similarity Index (FSIM) between a pair of images
    FSIM combines phase congruency and gradient magnitude information, better aligning with human visual perception
    """

    def __init__(self):
        super(FSIM, self).__init__()
        # Reflection padding, adding padding at image edges to avoid edge effects
        self.refl = nn.ReflectionPad2d(4)  # Larger padding needed for PC

        # Define Sobel operators for computing gradients
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)

        # Constant term for numerical stability
        self.T1 = 0.85
        self.T2 = 160
        self.eps = 1e-6

        # Filters needed for phase congruency (PC) computation
        self.pc_scales = 4  # Number of scales
        self.pc_orientations = 4  # Number of orientations
        self.pc_filters = self._create_pc_filters()

    def _create_pc_filters(self):
        """Create filter bank for computing phase congruency"""
        filters = []
        for scale in range(self.pc_scales):
            for orientation in range(self.pc_orientations):
                # Filter creation is simplified here; log-Gabor filters should be used in practice
                # A more complex filter design should be used in the actual implementation
                sigma = 0.5 * (1.0 + scale)
                wavelength = 4.0 * (1.0 + scale)
                theta = orientation * math.pi / self.pc_orientations

                # Create a simplified Gabor filter
                size = int(8 * sigma)
                if size % 2 == 0:
                    size += 1

                # This is just a simplified example; a more accurate Gabor filter implementation should be used
                filter_real = torch.zeros((size, size))
                filter_imag = torch.zeros((size, size))

                # In the actual implementation, fill in filter values
                # This is simplified; actual values should be computed based on the Gabor function

                filters.append((filter_real.reshape(1, 1, size, size),
                                filter_imag.reshape(1, 1, size, size)))
        return filters

    def _compute_phase_congruency(self, img):
        """Compute phase congruency of an image
        Note: This is a simplified implementation; actual PC computation is more complex
        """
        # In the actual implementation, a more complex phase congruency computation should be used
        # Here it serves only as a framework example

        # Pad the image
        img_pad = self.refl(img)

        # Initialize phase congruency map
        pc_map = torch.zeros_like(img)

        # In practice, the filter bank should be used to compute phase congruency
        # Here it is simplified to use gradient information instead
        sobel_x = self.sobel_x.to(img.device)
        sobel_y = self.sobel_y.to(img.device)

        grad_x = F.conv2d(img_pad, sobel_x, padding=0)
        grad_y = F.conv2d(img_pad, sobel_y, padding=0)

        # Use gradient magnitude as a simplified PC map
        pc_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.eps)

        return pc_map

    def forward(self, x, y):
        # Ensure input is a 4D tensor [batch, channels, height, width]
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if len(y.shape) == 3:
            y = y.unsqueeze(0)

        # If color image, convert to grayscale
        if x.shape[1] == 3:
            # RGB to grayscale conversion weights
            weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
            x_gray = (x * weights).sum(dim=1, keepdim=True)
            y_gray = (y * weights).sum(dim=1, keepdim=True)
        else:
            x_gray = x
            y_gray = y

        # Compute phase congruency (PC)
        pc_x = self._compute_phase_congruency(x_gray)
        pc_y = self._compute_phase_congruency(y_gray)

        # Compute gradient magnitudes
        x_pad = self.refl(x_gray)
        y_pad = self.refl(y_gray)

        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)

        grad_x_x = F.conv2d(x_pad, sobel_x, padding=0)
        grad_y_x = F.conv2d(x_pad, sobel_y, padding=0)
        grad_x_y = F.conv2d(y_pad, sobel_x, padding=0)
        grad_y_y = F.conv2d(y_pad, sobel_y, padding=0)

        grad_mag_x = torch.sqrt(grad_x_x ** 2 + grad_y_x ** 2 + self.eps)
        grad_mag_y = torch.sqrt(grad_x_y ** 2 + grad_y_y ** 2 + self.eps)

        # Compute gradient similarity
        G_similarity = (2 * grad_mag_x * grad_mag_y + self.T2) / (grad_mag_x ** 2 + grad_mag_y ** 2 + self.T2)

        # Compute PC similarity
        PC_similarity = (2 * pc_x * pc_y + self.T1) / (pc_x ** 2 + pc_y ** 2 + self.T1)

        # Compute FSIM
        numerator = PC_similarity * G_similarity
        PC_max = torch.maximum(pc_x, pc_y)

        fsim = torch.sum(numerator * PC_max, dim=[1, 2, 3]) / (torch.sum(PC_max, dim=[1, 2, 3]) + self.eps)

        # Convert to loss value
        #return torch.clamp(1 - fsim, 0, 1)

        return torch.clamp(1 - fsim, 0, 1).unsqueeze(1)

def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    eps = 1e-10
    rmse = torch.sqrt(rmse.mean()+eps)

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean()+eps)

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
