# -*- coding: utf-8 -*-
"""Single-image monocular depth inference for FlexDepth models.

Loads a trained encoder + depth decoder from a weights folder and predicts a
depth map for one image or every image in a folder.

Example:
    python inference.py --load_weights_folder ./models/kitti/flex_x \
        --encoder_model_type yolo11x-seg --decoder_model_type flexx \
        --image_path path/to/image.png --save_npy
"""
from __future__ import absolute_import, division, print_function

import glob
import os

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms

import networks
from layers import disp_to_depth
from options import FlexdepthOptions


def parse_args():
    opts = FlexdepthOptions()
    parser = opts.parser
    parser.add_argument("--image_path",
                        type=str,
                        required=True,
                        help="path to a single image or to a folder of images")
    parser.add_argument("--save_path",
                        type=str,
                        default=None,
                        help="where to save the colour depth map(s); defaults to "
                             "<image_dir>/depth/ when omitted")
    parser.add_argument("--save_npy",
                        action="store_true",
                        help="if set, also save the raw disparity map as a .npy file")
    return opts.parse()


def load_image(path, height, width):
    with open(path, "rb") as f:
        image = pil.open(f).convert("RGB")
    return image.resize((width, height), pil.LANCZOS)


def load_model(opt, device):
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path, map_location="cpu")
    encoder_dict = {k.replace("module.", ""): v for k, v in encoder_dict.items()}
    try:
        height, width = encoder_dict["height"], encoder_dict["width"]
        print("-> Using training resolution {}x{} from weights".format(width, height))
    except KeyError:
        height, width = opt.height, opt.width
        print("-> No height/width in weights, using {}x{}".format(width, height))

    enc_scale = opt.encoder_model_type.replace("yolo11", "")
    encoder = networks.YOLOEncoder(False, scale=enc_scale)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    dec_scale = opt.decoder_model_type.replace("flex", "")
    depth_decoder = networks.FlexDepthDecoder(
        encoder.num_ch_enc, opt.scales, scale=dec_scale, opt=opt)
    decoder_dict = {k.replace("module.", ""): v
                    for k, v in torch.load(decoder_path, map_location="cpu").items()}
    depth_model_dict = depth_decoder.state_dict()
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

    encoder.to(device).eval()
    depth_decoder.to(device).eval()
    return encoder, depth_decoder, height, width


def to_color_depth(disp, out_size):
    """Colourise a disparity map using the magma colormap, resized to out_size."""
    disp_resized = torch.nn.functional.interpolate(
        disp, out_size, mode="bilinear", align_corners=False)
    disp_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
    return (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)


def predict(encoder, depth_decoder, image, device):
    input_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = depth_decoder(encoder(input_tensor))
    return output[("disp", 0)]


def main():
    opt = parse_args()
    device = torch.device("cpu" if opt.no_cuda or not torch.cuda.is_available() else "cuda")

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a weights folder at {}".format(opt.load_weights_folder)
    print("-> Loading weights from {}".format(opt.load_weights_folder))
    encoder, depth_decoder, height, width = load_model(opt, device)

    if os.path.isdir(opt.image_path):
        in_dir = os.path.normpath(opt.image_path)
        image_files = sorted(
            glob.glob(os.path.join(in_dir, "*.[jp][pn]g"))
            + glob.glob(os.path.join(in_dir, "*.jpeg")))
        assert image_files, "No images found in {}".format(opt.image_path)
    else:
        assert os.path.isfile(opt.image_path), \
            "--image_path is neither a file nor a folder: {}".format(opt.image_path)
        in_dir = os.path.dirname(opt.image_path) or "."
        image_files = [opt.image_path]

    out_dir = opt.save_path or os.path.join(in_dir, "depth")
    os.makedirs(out_dir, exist_ok=True)
    print("-> Saving results to {}".format(out_dir))

    with torch.no_grad():
        for image_file in image_files:
            name = os.path.splitext(os.path.basename(image_file))[0]
            original = pil.open(image_file).convert("RGB")
            image = load_image(image_file, height, width)
            disp = predict(encoder, depth_decoder, image, device)

            colour = to_color_depth(disp, (original.height, original.width))
            pil.fromarray(colour).save(os.path.join(out_dir, name + "_depth.png"))

            if opt.save_npy:
                _, depth = disp_to_depth(disp, opt.min_depth, opt.max_depth)
                np.save(os.path.join(out_dir, name + "_disp.npy"),
                        disp.squeeze().cpu().numpy())
                np.save(os.path.join(out_dir, name + "_depth.npy"),
                        depth.squeeze().cpu().numpy())

            print("   Processed {}".format(image_file))

    print("-> Done. {} image(s) written to {}".format(len(image_files), out_dir))


if __name__ == "__main__":
    main()
