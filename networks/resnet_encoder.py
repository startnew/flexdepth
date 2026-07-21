import numpy as np
import sys

sys.path.append("../")
sys.path.append("./")

import torch
import torch.nn as nn
import torchvision.models
import torchvision.models as models

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        print(num_layers)
        # loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])

        if str(num_layers) == "18":
            # loaded = torchvision.models.resnet18(
            #     pretrained=pretrained)  # model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
            # todo 使用weights `weights=ResNet18_Weights.DEFAULT`   ResNet18_Weights.IMAGENET1K_V1
            loaded = torchvision.models.resnet18(
                pretrained=pretrained)
        elif str(num_layers) == "50":
            loaded = torchvision.models.resnet50(
                pretrained=pretrained)  # model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded = loaded.state_dict()

        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model
try:
    import ultralytics
except:
    ultralytics =lambda x:x
class YOLOEncoder(nn.Module):
    """Pytorch module  yolo v11 for  encoder
        """

    def __init__(self,  pretrained, num_input_images=1,scale="n-seg"):
        super(YOLOEncoder, self).__init__()
        from networks.yolo11 import YOLO11Encoder
        import os
        cfg_file = "./cfg/models/yolo11-dep-encoder.yaml"
        if os.path.exists(cfg_file):
            print("cfg file", cfg_file)
        else:
            import sys
            if sys.platform == "linux":
                cfg_file = "./cfg/models/yolo11-dep-encoder.yaml"
            else:
                print("Config file not found:", cfg_file)


            pass
        use_last = True
        model_encoder = YOLO11Encoder(cfg=cfg_file, ch=3*num_input_images,scale=scale.split("-")[0],use_last=use_last)
        self.encoder = model_encoder.model[:max(model_encoder.feature_idxs)+1]
        # self.initialize_weights(self.encoder)

        ckpt = None

        if pretrained and scale :
            pretrained = f"./ckpt/yolo11{scale}.pt"
            print("[Loading pretrained model] load model from ", pretrained)
            ckpt = torch.load(pretrained, map_location="cpu")


        self.feature_idxs = model_encoder.feature_idxs
        if ckpt is not None:
            ckpt["model"].model = ckpt["model"].model.float()
            ckpt["model"].model[0].conv.weight = torch.nn.Parameter(
            torch.cat([ckpt["model"].model[0].conv.weight] * num_input_images, 1) / num_input_images)

            model_dict = self.encoder.state_dict()
            # print(model_dict.keys())
            state_dict = ckpt["model"].model.state_dict()
            missing_keys = [k for k in model_dict.keys() if k not in state_dict]
            print(f"Missing keys: {missing_keys}")

            self.encoder.load_state_dict({k: v for k, v in ckpt["model"].model.state_dict().items() if k in model_dict.keys()},strict=False)
        #self.num_ch_enc = np.array([64, 64, 128, 256, 512])
            print("Weight loading complete, num_input_images:", num_input_images)

        self.num_ch_enc = []

        for i, x in enumerate(model_encoder.feature_chanel_idxs):
            if use_last:
                if i < 1 or i == len(model_encoder.feature_chanel_idxs) - 1:
                    self.num_ch_enc.append(self.encoder[x].conv.out_channels)
                else:
                    self.num_ch_enc.append(self.encoder[model_encoder.feature_chanel_idxs[i+1]].conv.in_channels)
            else:
                self.num_ch_enc.append(self.encoder[x].conv.out_channels)
                #self.num_ch_enc.append(self.encoder[model_encoder.feature_chanel_idxs[i + 1]].conv.in_channels)
        self.num_ch_enc = np.array(self.num_ch_enc)
        print("Encoder ,self.num_ch_enc{}".format(self.num_ch_enc))


    def initialize_weights(self,model):
        """Initialize model weights to random values."""
        print("Initializing weights")
        for m in model.modules():
            t = type(m)

            if t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
                m.inplace = True
            elif isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, ( nn.LayerNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def forward(self, input_image:torch.Tensor):
        self.features = []
        x = (input_image - 0.45) / 0.225
        for i,m in enumerate(self.encoder):
            x = m(x)
            if i in self.feature_idxs:
                self.features.append(x)

        return self.features



class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        # Only extract features, remove last two unused layers
#         print('''去除最后两层，为了DDP训练不报错 : If you already have done the above, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. Please include the loss function and the structure of the return value of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
        del self.encoder.fc

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):

        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        return self.features


