from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import FlexdepthOptions
import datasets
import networks
import time
import scipy.io
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)



def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


from utils_add.torch_utils import profile_once

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def align_disparity_lstsq(pred_disp, gt_depth, depth_cap):
    """LS alignment in disparity space: 1/d = shift + scale * output.

    Fits shift and scale so that (shift + scale * pred_disp) ≈ 1/gt_depth,
    then converts back to metric depth.
    pred_disp and gt_depth should be 1D arrays of valid pixels (already masked).
    Returns (metric_depth, scale, shift) so 2D visualization can reuse the params.
    """
    disp_pred = pred_disp
    disp_gt = 1.0 / gt_depth
    A = np.stack([disp_pred, np.ones_like(disp_pred)], axis=-1)
    scale, shift = np.linalg.lstsq(A, disp_gt, rcond=None)[0]
    pred_aligned_disp = shift + scale * pred_disp
    disparity_cap = 1.0 / depth_cap
    pred_aligned_disp = np.maximum(pred_aligned_disp, disparity_cap)
    metric_depth = 1.0 / pred_aligned_disp
    return metric_depth, scale, shift

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    warm = True
    num = 0
    times = []  # Per-batch times
    times_per_sample = []  # Per-sample times

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(opt.split_path, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)
        encoder_dict =  {k.replace('module.', ''): v for k, v in encoder_dict.items()}
        try:
            HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
        except KeyError:
            print('No "height" or "width" keys found in the encoder state_dict, resorting to '
                  'using command line values!')
            HEIGHT, WIDTH = opt.height, opt.width

        img_ext = '.png' if opt.png else '.jpg'
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "cityscapes_preprocessed": datasets.CityscapesEvalDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        print("【dataset】", opt.dataset, opt.split)
        # dataset = datasets_dict[opt.dataset]
        frames_to_load = [0]
        if opt.eval_split == 'cityscapes':
            dataset = datasets.CityscapesEvalDataset(opt.data_path, filenames,
                                                     HEIGHT, WIDTH,
                                                     frames_to_load, 4,
                                                     is_train=False, )

        else:
            dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                               encoder_dict['height'], encoder_dict['width'],
                                               frames_to_load, 4,
                                               is_train=False, img_ext=img_ext)

        # dataset = dataset(opt.data_path, filenames,
        #                   HEIGHT, WIDTH,
        #                                    [0], 4, is_train=False,img_ext=img_ext)
        print("dataset",dataset)
        # 16
        print("encoder_dict['height']",encoder_dict['height'],encoder_dict['width'])
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=False, drop_last=False)
        if opt.encoder_model_type == "resnet":
            encoder = networks.ResnetEncoder(
            opt.num_layers, False)


        elif "yolo11" in opt.encoder_model_type:
            print( "model scale:",opt.encoder_model_type.replace("yolo11",""))
            encoder = networks.YOLOEncoder(
                False,scale= opt.encoder_model_type.replace("yolo11",""))

        else:
            encoder = networks.ResnetEncoder(opt.num_layers, False)

        if opt.decoder_model_type == "ori":
            depth_decoder = networks.DepthDecoder(
                encoder.num_ch_enc, opt.scales)

        elif "flex" in opt.decoder_model_type:
            depth_decoder = networks.FlexDepthDecoder(
                encoder.num_ch_enc, opt.scales,
                scale=opt.decoder_model_type.replace("flex", ""),opt=opt)
        else:
            print("Not define Decoder ")

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        # print({k: v for k, v in encoder_dict.items() if k not in model_dict})

        decoder_dict =  {k.replace('module.', ''): v for k, v in torch.load(decoder_path).items()}
        depth_model_dict = depth_decoder.state_dict()
        depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()
        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))
        iterator = tqdm(dataloader)

        with torch.no_grad():

            for data in iterator:
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)


                # warm
                if num == 0 :
                    # Warm-up, run only once
                    #output = depth_decoder(encoder(input_color))
                    print(input_color.size)
                    flops, params, flops_e, params_e, flops_d, params_d = profile_once(encoder, depth_decoder,
                                                                                       input_color)
                num += 1
                # Measure time
                st =  time_sync()
                # print(f"input_color device: {input_color.device}")
                output = depth_decoder(encoder(input_color))

                ed = time_sync()
                times.append(ed - st)
                times_per_sample.append((ed - st )/ input_color.shape[0])


                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(opt.split_path, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]
    print("one batch avg interface use :",np.mean(times),"s\n","one sample avg use time",np.mean(times_per_sample)*1000,"ms\n","fps:",1/np.mean(times_per_sample),f"Num:{len(times)} use:{np.sum(times)}s")
    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()
    if opt.eval_split == 'cityscapes':
        print('loading cityscapes gt depths individually due to their combined size!')
        gt_depths = os.path.join(opt.split_path, opt.eval_split, "gt_depths")
    else:
        gt_path = os.path.join(opt.split_path, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    # gt_path = os.path.join(opt.split_path, opt.eval_split, "gt_depths.npz")
    # gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1',allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    lstsq_shifts=[]

    for i in range(pred_disps.shape[0]):

        if opt.eval_split == 'cityscapes':
            gt_depth = np.load(os.path.join(gt_depths, str(i).zfill(3) + '_depth.npy'))
            gt_height, gt_width = gt_depth.shape[:2]
            # crop ground truth to remove ego car -> this has happened in the dataloader for input
            # images
            gt_height = int(round(gt_height * 0.75))
            gt_depth = gt_depth[:gt_height]

        else:
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

        # gt_depth = gt_depths[i]
        # gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == 'cityscapes':
            # when evaluating cityscapes, we centre crop to the middle 50% of the image.
            # Bottom 25% has already been removed - so crop the sides and the top here
            gt_depth = gt_depth[256:, 192:1856]
            pred_depth = pred_depth[256:, 192:1856]

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        elif opt.eval_split == 'cityscapes':
            # Previously missing this crop
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            # when evaluating cityscapes, we centre crop to the middle 50% of the image.
            # Bottom 25% has already been removed - so crop the sides and the top here


        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if opt.use_lstsq_alignment:
            pred_depth_aligned, lstsq_scale, lstsq_shift = align_disparity_lstsq(
                1.0 / pred_depth, gt_depth, depth_cap=MAX_DEPTH)
            pred_depth = pred_depth_aligned
            ratios.append(lstsq_scale)
            lstsq_shifts.append(lstsq_shift)

        elif not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio


        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if opt.use_lstsq_alignment:
        ratios = np.array(ratios)
        lstsq_shifts = np.array(lstsq_shifts)
        med = np.median(ratios)
        print(" LSTSQ scales | med: {:0.6f} | std: {:0.6f}".format(med, np.std(ratios / med)))
        print(" LSTSQ shifts | med: {:0.6f} | std: {:0.6f}".format(np.median(lstsq_shifts), np.std(lstsq_shifts)))
    elif not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n  " + ("flops: {0}, params: {1}, flops_e: {2}, params_e:{3}, flops_d:{4}, params_d:{5}").format(flops,
                                                                                                             params,
                                                                                                             flops_e,
                                                                                                             params_e,
                                                                                                             flops_d,
                                                                                                             params_d))
    if opt.eval_split != 'cityscapes':
        eval_make_3d(opt.make3d_path,encoder=encoder,decoder=depth_decoder)
    print("\n-> Done!")


def eval_make_3d(main_path,encoder,decoder):
    # refer from https://github.com/nianticlabs/monodepth2/issues/392#issuecomment-1128977268
    import scipy
    def compute_3d_errors(gt, pred):
        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log10(gt) - np.log10(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        return abs_rel, sq_rel, rmse, rmse_log

    with open(os.path.join(main_path, "make3d_test_files.txt")) as f:
        test_filenames = f.read().splitlines()
    test_filenames = map(lambda x: x[4:-4], test_filenames)

    depths_gt = []
    images = []
    ratio = 2
    h_ratio = 1 / (1.33333 * ratio)
    color_new_height = 1704 // 2
    depth_new_height = 21

    for filename in test_filenames:
        mat = scipy.io.loadmat(os.path.join(main_path, "Gridlaserdata", "depth_sph_corr-{}.mat".format(filename)))
        depths_gt.append(mat["Position3DGrid"][:, :, 3])

        image = cv2.imread(os.path.join(main_path, "Test134", "img-{}.jpg".format(filename)))
        # print(depths_gt[-1].shape,"depths_gt")
        # print(image.shape,"image.shape")
        image = image[(2272 - color_new_height) // 2:(2272 + color_new_height) // 2, :]
        images.append(image[:, :, ::-1])
        # cv2.imwrite(os.path.join(main_path, "Test134_cropped", "img-{}.jpg".format(filename)), image)
    depths_gt_resized = map(lambda x: cv2.resize(x, (305, 407), interpolation=cv2.INTER_NEAREST), depths_gt)
    depths_gt_cropped = list(map(lambda x: x[(55 - 21) // 2:(55 + 21) // 2], depths_gt))


    # pred_disps = np.load(path_to_pred_disps)

    errors = []
    with torch.no_grad():
        for i in range(len(images)):
            input_color = images[i]
            input_color = cv2.resize(input_color / 255.0, (640, 192), interpolation=cv2.INTER_NEAREST)

            input_color = torch.tensor(input_color, dtype=torch.float).cuda().permute(2, 0, 1)[None, :, :, :]

            output = decoder(encoder(input_color))
            pred_disp, _ = disp_to_depth(output[("disp", 0)], 0.1, 100)
            pred_disp = pred_disp.squeeze().cpu().numpy()
            depth_gt = depths_gt_cropped[i]
            depth_pred = 1 / pred_disp
            depth_pred = cv2.resize(depth_pred, depth_gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
            mask = np.logical_and(depth_gt > 0, depth_gt < 70)
            depth_gt = depth_gt[mask]
            depth_pred = depth_pred[mask]
            depth_pred *= np.median(depth_gt) / np.median(depth_pred)
            depth_pred[depth_pred > 70] = 70
            errors.append(compute_3d_errors(depth_gt, depth_pred))
        mean_errors = np.mean(errors, 0)
    print("Evaluate Make3d:")
    print(("{:>8} | " * 4).format("abs_rel", "sq_rel", "rmse", "rmse_log"))
    print(("{: 8.3f} , " * 4).format(*mean_errors.tolist()))

if __name__ == "__main__":
    options = FlexdepthOptions()
    evaluate(options.parse())