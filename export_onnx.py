#

from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
from tqdm import tqdm
import torch

import torch.nn as nn
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import FlexdepthOptions
import datasets
import networks

import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from torchvision import transforms, datasets

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

def test_one(model,image_path,feed_width,feed_height):
    input_image = pil.open(image_path).convert('RGB')
    original_width, original_height = input_image.size
    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    # PREDICTION
    # if torch.cuda.is_available() and not args.no_cuda:
    #     device = torch.device("cuda")
    # else:
    device = torch.device("cpu")
    input_image = input_image.to(device)
    print(input_image.max(),input_image.min())
    outputs = model(input_image)

    #print(outputs)

    disp = outputs#[0]
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)

    # Saving numpy file
    output_name = os.path.splitext(os.path.basename(image_path))[0]

    scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    print("disp_resized_np",disp_resized_np.shape)
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)


    im.show()

def export_onnx(opt):
    onnx_output_path = "./models"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)
        encoder_dict = {k.replace('module.', ''): v for k, v in encoder_dict.items()}

        try:
            HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
        except KeyError:
            print('No "height" or "width" keys found in the encoder state_dict, resorting to '
                  'using command line values!')
            HEIGHT, WIDTH = opt.height, opt.width
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
            depth_decoder = networks.DepthDecoderExport(
                encoder.num_ch_enc, opt.scales)

        elif "flex" in opt.decoder_model_type:
            if max(opt.scales) == 2:
                raise NotImplementedError(
                    "ONNX export with --scales 3 is not yet supported. Use --scales 4 instead.")
            else:
                if opt.upsample in ["dysample_pl", "dysample_lp"] and not opt.c3k:
                    print("use dysample","FlexDepthDecoderLiteScale4DyPLExport")
                    depth_decoder = networks.FlexDepthDecoderLiteScale4DyPLExport(
                        encoder.num_ch_enc, opt.scales,
                        scale=opt.decoder_model_type.replace("flex", ""), opt=opt)
                elif opt.c3k and opt.upsample == "dysample_pl" and opt.dysample_group8 and opt.all_dysample:
                    print(opt.decoder_model_type.replace("flex", ""))
                    print("scale FlexDepthDecoderLiteScale4DyPLC3KG8Export")
                    depth_decoder = networks.FlexDepthDecoderLiteScale4DyPLC3KG8Export(
                        encoder.num_ch_enc, opt.scales,
                        scale=opt.decoder_model_type.replace("flex", ""), opt=opt)
                else:
                    print("scale FlexDepthDecoderLiteScale4Export")
                    depth_decoder = networks.FlexDepthDecoderLiteScale4Export(
                    encoder.num_ch_enc, opt.scales,
                    scale=opt.decoder_model_type.replace("flex", ""), opt=opt)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

        decoder_dict =  {k.replace('module.', ''): v for k, v in torch.load(decoder_path).items()}
        depth_model_dict = depth_decoder.state_dict()
        depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})
        # not load params
        #print({k: v for k, v in decoder_dict.items() if k not in depth_model_dict})
        # depth_decoder.load_state_dict(decoder_dict)

        #encoder.cuda()
        class Monodepth2(nn.Module):
            def __init__(self, encoder, decoder):
                super(Monodepth2, self).__init__()
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, x):
                features = self.encoder(x)
                outputs = self.decoder(features)
                return outputs

        encoder.eval()
        #depth_decoder.cuda()
        depth_decoder.eval()
        use_union = True
        dummy_input = torch.randn(1, 3, HEIGHT, WIDTH, requires_grad=True)
        if use_union:
            full_model = Monodepth2(encoder, depth_decoder)
            #导出完整的模型
            full_model.eval()

            full_model_onnx_path = f"{onnx_output_path}/{opt.export_name}.onnx"

            output_names = ['disp_0']
            torch.onnx.export(full_model,
                              dummy_input,
                              full_model_onnx_path,
                              export_params=True,
                              # opset_version=16
                              opset_version=16,
                              do_constant_folding=True,
                              input_names=['input'],
                              output_names=output_names,
                              dynamic_axes={'input': {0: 'batch_size'},
                                            'disp_0': {0: 'batch_size'}}
                           )
            with torch.no_grad():
                results = full_model(dummy_input)
                print(results[0].shape,len(results))

            print("encoder export ok")

            print("\nCombined ONNX export complete.")
        else:
            # --- 导出 ONNX ---
            # 创建一个符合模型输入的虚拟输入张量
            # HEIGHT, WIDTH
            dummy_input = torch.randn(1, 3, HEIGHT, WIDTH, requires_grad=True)

            print(f"\n-> Starting ONNX export to {onnx_output_path}")

            # PyTorch 的 onnx.export 函数
            # 注意：这里我们将 encoder 和 decoder 一起导出。
            # 在实际使用中，我们需要先运行 encoder，再将结果输入 decoder。
            # 为了简化部署，我们可以在一个模型中完成。
            # 但 Monodepth2 的结构直接导出不方便，我们这里只导出 encoder 和 decoder 分开的模型，在 ROS 节点中依次调用。
            # 更优化的方法是创建一个包含 encoder 和 decoder 的新 nn.Module，然后导出。

            # 导出 Encoder
            encoder_onnx_path = f"{onnx_output_path}/flexdepth_encoder.onnx"
            print(f"-> Exporting encoder to {encoder_onnx_path}")
            torch.onnx.export(encoder,
                              dummy_input,
                              encoder_onnx_path,
                              export_params=True,
                              opset_version=16,
                              do_constant_folding=True,
                              input_names=['input'],
                              output_names=['features_0', 'features_1', 'features_2', 'features_3', 'features_4'],
                              dynamic_axes={'input': {0: 'batch_size'},
                                            'features_0': {0: 'batch_size'},
                                            'features_1': {0: 'batch_size'},
                                            'features_2': {0: 'batch_size'},
                                            'features_3': {0: 'batch_size'},
                                            'features_4': {0: 'batch_size'}})

            # 导出 Decoder
            # 创建 decoder 的虚拟输入
            with torch.no_grad():
                dummy_features = encoder(dummy_input)

            print("encoder export ok")

            decoder_onnx_path =  f"{onnx_output_path}/flexdepth_decoder.onnx"
            print(f"-> Exporting decoder to {decoder_onnx_path}")
            if max(opt.scales) == 3:
                output_names = ['disp_0', 'disp_1', 'disp_2', 'disp_3']
            elif max(opt.scales) == 2:
                output_names = ['disp_0', 'disp_1', 'disp_2', ]
            print(len(dummy_features),dummy_features[0].shape)
            print(output_names)

            torch.onnx.export(depth_decoder,
                              dummy_features,
                              decoder_onnx_path,
                              export_params=True,
                              opset_version=16,
                              do_constant_folding=True,
                              input_names=['features_0', 'features_1', 'features_2', 'features_3', 'features_4'],
                              output_names=output_names,
                              dynamic_axes={'features_0': {0: 'batch_size'},
                                            'features_1': {0: 'batch_size'},
                                            'features_2': {0: 'batch_size'},
                                            'features_3': {0: 'batch_size'},
                                            'features_4': {0: 'batch_size'},
                                            'disp_0': {0: 'batch_size'}})
            with torch.no_grad():
                result = depth_decoder(dummy_features)
                print(result[0].shape)

            print("\nONNX export complete.")


if __name__ == "__main__":
    options = FlexdepthOptions()
    export_onnx(options.parse())

    # Flex-Nano ONNX export (auto-defaults apply):
    # python export_onnx.py --encoder_model_type yolo11n-seg --decoder_model_type flexn --load_weights_folder ./models/kitti/flex_n/weights_41 --scales 4 --export_name flex-n

    # Flex-X-Large ONNX export (auto-defaults apply):
    # python export_onnx.py --encoder_model_type yolo11x-seg --decoder_model_type flexx --load_weights_folder ./models/kitti/flex_x/weights_38 --scales 4 --export_name flex-x
