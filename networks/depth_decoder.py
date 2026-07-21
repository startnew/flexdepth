# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
# from layers import *
from layers import ConvBlock,Conv3x3,upsample,DySample
from utils_add.conv import Conv
import math
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import torch.nn.init as init # 引入初始化工具
class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, freeze=0):
        super(DepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = "nearest"#'bilinear'#'nearest'
        self.scales = scales
        self.freeze = freeze
        print("freeze",freeze)

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.num_ch_dec = [int(x) for x in self.num_ch_dec]

        # decoder
        self.convs = OrderedDict()

        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_features):
        self.outputs = {}
        #print("【self.freeze】，self.freeze",self.freeze)

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x,mode=self.upsample_mode)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                if self.freeze == 1:
                    self.outputs[("disp_f", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                elif self.freeze == 2:
                    self.outputs[("disp_f_0", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                else:
                    self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

class DepthDecoderExport(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, freeze=0):
        super(DepthDecoderExport, self).__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = "nearest"#'bilinear'#'nearest'
        self.scales = scales
        self.freeze = freeze
        print("freeze",freeze)

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.num_ch_dec = [int(x) for x in self.num_ch_dec]

        # decoder
        self.convs = OrderedDict()

        # i = 4 (Level 4 -> 3)
        # upconv_0: num_ch_in = self.num_ch_enc[-1] (512), num_ch_out = self.num_ch_dec[4] (256)
        self.convs[("upconv", 4, 0)] = ConvBlock(512, 256)

        # [BUG FIX]
        # upconv_1:
        #   num_ch_in = self.num_ch_dec[4] (256)
        #   + skip (i=4 > 0 is True): self.num_ch_enc[3] (256)
        #   = 512
        #   num_ch_out = self.num_ch_dec[4] (256)
        self.convs[("upconv", 4, 1)] = ConvBlock(512, 256)

        # i = 3 (Level 3 -> 2)
        # upconv_0: num_ch_in = self.num_ch_dec[4] (256), num_ch_out = self.num_ch_dec[3] (128)
        self.convs[("upconv", 3, 0)] = ConvBlock(256, 128)
        # upconv_1: num_ch_in = self.num_ch_dec[3] (128) + skip: self.num_ch_enc[2] (128) -> 256
        self.convs[("upconv", 3, 1)] = ConvBlock(256, 128)

        # i = 2 (Level 2 -> 1)
        # upconv_0: num_ch_in = self.num_ch_dec[3] (128), num_ch_out = self.num_ch_dec[2] (64)
        self.convs[("upconv", 2, 0)] = ConvBlock(128, 64)
        # upconv_1: num_ch_in = self.num_ch_dec[2] (64) + skip: self.num_ch_enc[1] (64) -> 128
        self.convs[("upconv", 2, 1)] = ConvBlock(128, 64)

        # i = 1 (Level 1 -> 0)
        # upconv_0: num_ch_in = self.num_ch_dec[2] (64), num_ch_out = self.num_ch_dec[1] (32)
        self.convs[("upconv", 1, 0)] = ConvBlock(64, 32)
        # upconv_1: num_ch_in = self.num_ch_dec[1] (32) + skip: self.num_ch_enc[0] (64) -> 96
        self.convs[("upconv", 1, 1)] = ConvBlock(96, 32)

        # i = 0 (Level 0 -> Output)
        # upconv_0: num_ch_in = self.num_ch_dec[1] (32), num_ch_out = self.num_ch_dec[0] (16)
        self.convs[("upconv", 0, 0)] = ConvBlock(32, 16)
        # upconv_1: num_ch_in = self.num_ch_dec[0] (16) + skip (i=0 is False) -> 16
        self.convs[("upconv", 0, 1)] = ConvBlock(16, 16)

        # for s in self.scales 完全展开 (s = 0, 1, 2, 3)
        self.convs[("dispconv", 0)] = Conv3x3(self.num_ch_dec[0], 1)  # 16 channels
        self.convs[("dispconv", 1)] = Conv3x3(self.num_ch_dec[1], 1)  # 32 channels
        self.convs[("dispconv", 2)] = Conv3x3(self.num_ch_dec[2], 1)  # 64 channels
        self.convs[("dispconv", 3)] = Conv3x3(self.num_ch_dec[3], 1)  # 128 channels

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_features):
        x = input_features[4]  # x = input_features[-1]

        # i = 4
        x = self.convs[("upconv", 4, 0)](x)
        x = [upsample(x, mode=self.upsample_mode)]
        # [BUG FIX] if self.use_skips and i > 0: True (i=4)
        x += [input_features[3]]  # input_features[i - 1]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 4, 1)](x)
        # if i in self.scales: False (i=4)

        # i = 3
        x = self.convs[("upconv", 3, 0)](x)
        x = [upsample(x, mode=self.upsample_mode)]
        # if self.use_skips and i > 0: True
        x += [input_features[2]]  # input_features[i - 1]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 3, 1)](x)
        # if i in self.scales: True (i=3)
        # Since self.freeze == 0, we use 'else' block
        disp3 = self.sigmoid(self.convs[("dispconv", 3)](x))

        # i = 2
        x = self.convs[("upconv", 2, 0)](x)
        x = [upsample(x, mode=self.upsample_mode)]
        # if self.use_skips and i > 0: True
        x += [input_features[1]]  # input_features[i - 1]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 2, 1)](x)
        # if i in self.scales: True (i=2)
        # Since self.freeze == 0, we use 'else' block
        disp2 = self.sigmoid(self.convs[("dispconv", 2)](x))

        # i = 1
        x = self.convs[("upconv", 1, 0)](x)
        x = [upsample(x, mode=self.upsample_mode)]
        # if self.use_skips and i > 0: True
        x += [input_features[0]]  # input_features[i - 1]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 1, 1)](x)
        # if i in self.scales: True (i=1)
        # Since self.freeze == 0, we use 'else' block
        disp1 = self.sigmoid(self.convs[("dispconv", 1)](x))

        # i = 0
        x = self.convs[("upconv", 0, 0)](x)
        x = [upsample(x, mode=self.upsample_mode)]
        # if self.use_skips and i > 0: False (i=0)
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 0, 1)](x)
        # if i in self.scales: True (i=0)
        # Since self.freeze == 0, we use 'else' block
        disp0 = self.sigmoid(self.convs[("dispconv", 0)](x))
        # 按照 ONNX 惯例，直接返回张量元组
        return disp0

from utils_add.block import  C3k2


class FlexDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, scale="n", freeze=0,opt=None):
        super(FlexDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        # self.upsample_mode = "nearest"#"bilinear"  # 'nearest'
        self.upsample_mode = "nearest"  # "bilinear"#'nearest'
        if opt is not None:
            if opt.upsample == "nearest":
                self.upsample_mode = "nearest"
            elif opt.upsample == "bilinear":
                self.upsample_mode = "bilinear"
            elif opt.upsample == "dysample_pl":
                self.upsample_mode = "dysample_pl"
            elif opt.upsample == "dysample_lp":
                self.upsample_mode = "dysample_lp"



        print("self.upsample_mode 【】", self.upsample_mode)
        self.scales = scales
        if opt.dy_mu:
            print("Using dynamic mu")

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([32, 64, 128, 256, 512])
        # max_channels = float("inf")
        max_channels = float("inf")
        model_scales = {'n': [0.5, 0.25, 1024], 's': [0.5, 0.5, 1024], 'm': [0.5, 1.0, 512], 'l': [1.0, 1.0, 512],
                        'x': [1.0, 1.5, 512]}
        depth, width, max_channels = model_scales[scale]
        # n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        self.num_out_mix = 4  # max(self.scales)
        if self.upsample_mode in ["dysample_pl","dysample_lp"]:
            self.use_all_upsample_learn = True
            self.use_dysample = True
        else:
            self.use_all_upsample_learn = False
        if self.use_all_upsample_learn:
            print("Setting all upsampling to learnable")


        # self.num_ch_dec = np.array([64, 128, 256, 512, 1024])  #
        self.num_ch_dec = np.array([int(min(x * width , max_channels)) for x in self.num_ch_dec])
        self.num_ch_dec = [int(x) for x in self.num_ch_dec]
        print("num_ch_dec", self.num_ch_dec)
        self.use_tiny = True
        # scales 4 时
        if opt.dysample_group8:
            group = 8
            print("dysample group use 8")
        else:
            group = 4
        if max(self.scales) == 3:
            self.tiny_num = 1
        elif max(self.scales) == 2:
            # scales 3 时
            self.tiny_num = 2

        self.freeze = freeze

        # decoder
        self.convs = OrderedDict()
        for i in range(self.num_out_mix, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == self.num_out_mix else self.num_ch_dec[i + 1]

            num_ch_out = self.num_ch_dec[i]
            if self.use_all_upsample_learn:

                if self.upsample_mode == "dysample_pl":
                    if opt.dysample_dynamic:
                        print("dysample use dyscope")
                        self.convs[("upconv_upsample", i, 0)] = DySample(num_ch_in,style="pl",dyscope=True,groups=group)
                    else:
                        self.convs[("upconv_upsample", i, 0)] = DySample(num_ch_in, style="pl",groups=group)

                elif self.upsample_mode == "dysample_lp":
                    if opt.dysample_dynamic:
                        self.convs[("upconv_upsample", i, 0)] = DySample(num_ch_in, style="lp",dyscope=True,groups=group)
                    else:
                        self.convs[("upconv_upsample", i, 0)] = DySample(num_ch_in, style="lp",groups=group)
                else:
                    self.convs[("upconv_upsample", i, 0)] = nn.ConvTranspose2d(num_ch_in, num_ch_out, 2,

                                                                               2, 0, bias=True)
                    num_ch_in = self.num_ch_dec[i]
            # num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            if opt is not None:
                if opt.c3mixk:
                    print("use Mix C3k")
                    if i == 1:
                        print("use c3k")
                        self.convs[("upconv", i, 1)] = C3k2(num_ch_in, num_ch_out,
                                                            c3k=True)  # ConvBlock(num_ch_in, num_ch_out)
                    else:
                        print("use c2f")
                        self.convs[("upconv", i, 1)] = C3k2(num_ch_in, num_ch_out,
                                                            c3k=False)  # ConvBlock(num_ch_in, num_ch_out)



                    pass
                else:
                    if opt.c3k:
                        print("use C3K")
                        self.convs[("upconv", i, 1)] = C3k2(num_ch_in, num_ch_out,c3k=True)  # ConvBlock(num_ch_in, num_ch_out)
                    else:
                        self.convs[("upconv", i, 1)] = C3k2(num_ch_in, num_ch_out)  # ConvBlock(num_ch_in, num_ch_out)
            else:
                self.convs[("upconv", i, 1)] = C3k2(num_ch_in, num_ch_out)  # ConvBlock(num_ch_in, num_ch_out)
            if self.use_tiny:
                if i < self.tiny_num + 1:
                    break

        for s in self.scales:
            if self.use_tiny:
                s = s + self.tiny_num
            if self.use_all_upsample_learn and self.upsample_mode not in ["dysample_lp","dysample_pl"]:
                for i in range(self.tiny_num):
                    if i == 0:
                        self.convs[(f"upsample_with_params_{i}", s)] = nn.ConvTranspose2d(self.num_ch_dec[s],
                                                                                          self.num_output_channels, 2,
                                                                                          2, 0, bias=True)
                    else:
                        self.convs[(f"upsample_with_params_{i}", s)] = nn.ConvTranspose2d(self.num_output_channels,
                                                                                          self.num_output_channels, 2,
                                                                                          2, 0,
                                                                                          bias=True)
            elif opt.all_dysample and self.upsample_mode  in ["dysample_lp","dysample_pl"]:
                print("use all_dysample",opt.all_dysample)
                self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s],
                                                      self.num_output_channels)  # Conv3x3(self.num_ch_dec[s], self.num_output_channels)
                if not opt.fuse_dysample :
                    for i in range(self.tiny_num):
                        if self.upsample_mode == "dysample_pl":
                            if opt.dysample_dynamic:
                                self.convs[(f"upsample_with_params_{i}", s)] = DySample(self.num_ch_dec[s],style="pl",dyscope=True,groups=group)
                            else:
                                self.convs[(f"upsample_with_params_{i}", s)] = DySample(self.num_ch_dec[s], style="pl",groups=group)
                        elif self.upsample_mode == "dysample_lp":
                            if opt.dysample_dynamic:
                                self.convs[(f"upsample_with_params_{i}", s)] = DySample(self.num_ch_dec[s], style="lp",dyscope=True,groups=group)
                            else:
                                self.convs[(f"upsample_with_params_{i}", s)] = DySample(self.num_ch_dec[s], style="lp",groups=group)
                else:
                    if s == 1:
                        for i in range(self.tiny_num):
                            if self.upsample_mode == "dysample_pl":
                                if opt.dysample_dynamic:
                                    self.convs[(f"upsample_with_params_{i}", s)] = DySample(self.num_ch_dec[s],
                                                                                            style="pl", dyscope=True,
                                                                                            groups=group)
                                else:
                                    self.convs[(f"upsample_with_params_{i}", s)] = DySample(self.num_ch_dec[s],
                                                                                            style="pl", groups=group)
                            elif self.upsample_mode == "dysample_lp":
                                if opt.dysample_dynamic:
                                    self.convs[(f"upsample_with_params_{i}", s)] = DySample(self.num_ch_dec[s],
                                                                                            style="lp", dyscope=True,
                                                                                            groups=group)
                                else:
                                    self.convs[(f"upsample_with_params_{i}", s)] = DySample(self.num_ch_dec[s],
                                                                                            style="lp", groups=group)

                    print("Using reusable upsampling module for further model simplification")

            else:

                self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s],
                                                  self.num_output_channels)  # Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)
        self.outputs = {}
        self.opt = opt
        print(self.convs.keys())




    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self,input_features):
        if self.opt.fuse_dysample:
            return  self.forward_fuse_dysample(input_features)
        else:
            return self.forward_base(input_features)

    def forward_fuse_dysample(self,input_features):
        # 使用可重复利用的upsample
        outputs = {}
        # decoder
        x = input_features[self.num_out_mix]  # input_features[-1]
        # 增加返回用于预测超参的特征
        for i in range(self.num_out_mix, -1, -1):
            if self.use_tiny:
                ind = i - self.tiny_num
                if ind < 0:
                    break
            else:
                ind = i
            # 之前先上采样再预测
            # 现在直接concat 再预测
            if i == self.num_out_mix:
                if self.use_all_upsample_learn:
                    if self.upsample_mode in ["dysample_pl", "dysample_lp"]:
                        x = [self.convs[("upconv_upsample", i, 0)](x)]
                    else:
                        x = [self.convs[("upconv_upsample", i, 0)](x)]

                else:
                    x = [upsample(x, mode=self.upsample_mode)]
            else:
                # 之前已经上采样了
                x = [x]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]

            # 特征融合
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            # 上采样
            if self.use_all_upsample_learn:
                if i > 1:
                    if self.upsample_mode in ["dysample_pl", "dysample_lp"]:
                        x = self.convs[("upconv_upsample", i-1, 0)](x)

                    else:
                        x = self.convs[("upconv_upsample", i-1, 0)](x)
                else:
                    for j in range(self.tiny_num):
                        x = self.convs[(f"upsample_with_params_{j}", i)](x)


            else:
                x = upsample(x, mode=self.upsample_mode)

            f = self.convs[("dispconv", i)](x)
            if self.freeze == 1:
                outputs[("disp_f", ind)] = self.sigmoid(f)  # self.sigmoid(self.convs[("dispconv", i)](x))
            elif self.freeze == 2:
                outputs[("disp_f_0", ind)] = self.sigmoid(f)
            else:
                outputs[("disp", ind)] = self.sigmoid(f)
        # print(outputs.keys())
        self.outputs = outputs

        return self.outputs

    def forward_base(self, input_features):
        outputs = {}
        # decoder
        x = input_features[self.num_out_mix]  # input_features[-1]
        for i in range(self.num_out_mix, -1, -1):

            if self.use_tiny:
                ind = i - self.tiny_num
                if ind < 0:
                    break
            else:
                ind = i

            # x = self.convs[("upconv", i, 0)](x)
            if self.use_all_upsample_learn:
                if self.upsample_mode in ["dysample_pl","dysample_lp"]:
                    x = [self.convs[("upconv_upsample", i, 0)](x)]

                else:
                    x = [self.convs[("upconv_upsample", i, 0)](x)]

            else:
                x = [upsample(x,mode=self.upsample_mode)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]

            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            #  x = self.convs[("upconv", i, 0)](x)
            #             x = [upsample(x)]
            #             if self.use_skips and i > 0:
            #                 x += [input_features[i - 1]]
            #             x = torch.cat(x, 1)n

            #             x = self.convs[("upconv", i, 1)](x)
            #             if i in self.scales:
            #                 self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

            if ind in self.scales:
                # f = upsample(self.convs[("dispconv", i)](x), mode='bilinear')
                if self.use_tiny:
                    if not self.use_all_upsample_learn :
                        f = self.convs[("dispconv", i)](x)
                        for j in range(self.tiny_num):
                            f = upsample(f, mode='bilinear')# bilinear
                    else:

                        if self.upsample_mode in ["dysample_pl","dysample_lp"]:
                            if self.opt.all_dysample:
                                if self.opt.fuse_dysample:
                                    pass
                                else:
                                    for j in range(self.tiny_num):
                                        f = self.convs[(f"upsample_with_params_{j}", i)](x)
                                    f = self.convs[("dispconv", i)](f)
                            else:

                                f = self.convs[("dispconv", i)](x)
                                for j in range(self.tiny_num):
                                    f = upsample(f, mode='bilinear')  # bilinear
                        else:
                            f = x
                            for j in range(self.tiny_num):
                                f = self.convs[(f"upsample_with_params_{j}", i)](f)

                else:
                    f = self.convs[("dispconv", ind)](x)

                if self.freeze == 1:
                    if ind == 0 and self.opt.dy_mu:
                        # 增加用于预测的特征图的返回
                        outputs["feature_disp"] = x

                    outputs[("disp_f", ind)] = self.sigmoid(f)  # self.sigmoid(self.convs[("dispconv", i)](x))
                elif self.freeze == 2:
                    outputs[("disp_f_0", ind)] = self.sigmoid(f)
                    if ind == 0 and self.opt.dy_mu:
                        # 增加用于预测的特征图的返回
                        outputs["feature_disp_0"] = x
                else:
                    outputs[("disp", ind)] = self.sigmoid(f)
        self.outputs = outputs

        return self.outputs



class FlexDepthDecoderLiteScale4Export(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, scale="n", freeze=0,opt=None):
        super(FlexDepthDecoderLiteScale4Export, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips

        self.upsample_mode = "nearest"  # "bilinear"#'nearest'
        if opt is not None:
            if opt.upsample == "nearest":
                self.upsample_mode = "nearest"
            elif opt.upsample == "bilinear":
                self.upsample_mode = "bilinear"

        if self.upsample_mode in ["dysample_pl", "dysample_lp"]:
            self.use_all_upsample_learn = True
            self.use_dysample = True
        else:
            self.use_all_upsample_learn = False

        print("self.upsample_mode 【】", self.upsample_mode)
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([32, 64, 128, 256, 512])


        model_scales = {'n': [0.5, 0.25, 1024], 's': [0.5, 0.5, 1024], 'm': [0.5, 1.0, 512], 'l': [1.0, 1.0, 512],
                        'x': [1.0, 1.5, 512]}
        depth, width, max_channels = model_scales[scale]

        self.num_out_mix = 4  # max(self.scales)
        self.num_ch_dec = np.array([int(min(x * width , max_channels)) for x in self.num_ch_dec])
        self.num_ch_dec = [int(x) for x in self.num_ch_dec]
        print("num_ch_dec", self.num_ch_dec)




        self.freeze = freeze

        # decoder
        # self.convs = OrderedDict()
        self.convs = OrderedDict()
        num_ch_in = self.num_ch_enc[-1]
        num_ch_out = self.num_ch_dec[4]
        num_ch_in += self.num_ch_enc[3]





        self.convs[("upconv",4,1)] = C3k2(num_ch_in, num_ch_out)
        num_ch_in = num_ch_out + self.num_ch_enc[2]
        num_ch_out = self.num_ch_dec[3]



        self.convs[("upconv",3,1)] = C3k2(num_ch_in, num_ch_out)
        num_ch_in = num_ch_out + self.num_ch_enc[1]
        num_ch_out = self.num_ch_dec[2]

        self.convs[("upconv",2,1)] = C3k2(num_ch_in, num_ch_out)



        num_ch_in = num_ch_out + self.num_ch_enc[0]
        num_ch_out = self.num_ch_dec[1]


        self.convs[("upconv", 1, 1)] = C3k2(num_ch_in, num_ch_out)

        self.convs[("dispconv", 1)] = Conv3x3(self.num_ch_dec[1],
                                              self.num_output_channels)

        self.convs[("dispconv",2)] = Conv3x3(self.num_ch_dec[2],
                                              self.num_output_channels)
        self.convs[("dispconv", 3)] = Conv3x3(self.num_ch_dec[3],
                                              self.num_output_channels)

        self.convs[("dispconv", 4)] = Conv3x3(self.num_ch_dec[4],
                                              self.num_output_channels)


        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)
        self.outputs = {}

        self.scale_factor = 2.0

    def upsample(self, x:torch.Tensor,mode="nearest"):
        # 使用实例属性
        return F.interpolate(x, scale_factor=self.scale_factor, mode=mode)



    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        outputs = {}
        # decoder
        x = input_features[self.num_out_mix]  # input_features[-1]


        x = [self.upsample(x, mode=self.upsample_mode)]
        x += [input_features[3]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv",4,1)](x)

        f = self.convs[("dispconv", 4)](x)
        f = self.upsample(f, mode='bilinear')


        outputs[("disp", 3)] = self.sigmoid(f)
        x = [upsample(x, mode=self.upsample_mode)]
        x += [input_features[2]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 3, 1)](x)
        f = self.convs[("dispconv", 3)](x)
        f = self.upsample(f, mode='bilinear')


        outputs[("disp", 2)] = self.sigmoid(f)

        x = [upsample(x, mode=self.upsample_mode)]
        x += [input_features[1]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 2, 1)](x)
        f = self.convs[("dispconv", 2)](x)
        f = self.upsample(f, mode='bilinear')


        outputs[("disp", 1)] = self.sigmoid(f)

        x = [upsample(x, mode=self.upsample_mode)]
        x += [input_features[0]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 1, 1)](x)
        f = self.convs[("dispconv", 1)](x)
        f = self.upsample(f, mode='bilinear')


        outputs[("disp", 0)] = self.sigmoid(f)


        return outputs[("disp", 0)]


class FlexDepthDecoderLiteScale4DyPLExport(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, scale="n", freeze=0,
                 opt=None):
        super(FlexDepthDecoderLiteScale4DyPLExport, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips

        self.upsample_mode = "nearest"  # "bilinear"#'nearest'
        if opt is not None:
            if opt.upsample == "nearest":
                self.upsample_mode = "nearest"
            elif opt.upsample == "bilinear":
                self.upsample_mode = "bilinear"
            elif opt.upsample == "dysample_pl":
                self.upsample_mode = "dysample_pl"
            elif opt.upsample == "dysample_lp":
                self.upsample_mode = "dysample_lp"

        if self.upsample_mode in ["dysample_pl", "dysample_lp"]:
            self.use_all_upsample_learn = True
            self.use_dysample = True
        else:
            self.use_all_upsample_learn = False

        print("self.upsample_mode 【】", self.upsample_mode)
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([32, 64, 128, 256, 512])

        model_scales = {'n': [0.5, 0.25, 1024], 's': [0.5, 0.5, 1024], 'm': [0.5, 1.0, 512], 'l': [1.0, 1.0, 512],
                        'x': [1.0, 1.5, 512]}
        depth, width, max_channels = model_scales[scale]

        self.num_out_mix = 4  # max(self.scales)
        self.num_ch_dec = np.array([int(min(x * width, max_channels)) for x in self.num_ch_dec])
        self.num_ch_dec = [int(x) for x in self.num_ch_dec]
        print("num_ch_dec", self.num_ch_dec)

        self.freeze = freeze

        # decoder
        # self.convs = OrderedDict()
        self.convs = OrderedDict()
        num_ch_in = self.num_ch_enc[-1]
        if self.upsample_mode == "dysample_pl":
            self.convs[("upconv_upsample", 4, 0)] = DySample(num_ch_in, style="pl")
        elif self.upsample_mode == "dysample_lp":
            self.convs[("upconv_upsample", 4, 0)] = DySample(num_ch_in, style="lp")

        num_ch_out = self.num_ch_dec[4]
        num_ch_in += self.num_ch_enc[3]



        self.convs[("upconv", 4, 1)] = C3k2(num_ch_in, num_ch_out)
        if self.upsample_mode == "dysample_pl":
            self.convs[("upconv_upsample", 3, 0)] = DySample(num_ch_out, style="pl")
        elif self.upsample_mode == "dysample_lp":
            self.convs[("upconv_upsample", 3, 0)] = DySample(num_ch_out, style="lp")
        num_ch_in = num_ch_out + self.num_ch_enc[2]
        num_ch_out = self.num_ch_dec[3]



        self.convs[("upconv", 3, 1)] = C3k2(num_ch_in, num_ch_out)
        if self.upsample_mode == "dysample_pl":
            self.convs[("upconv_upsample", 2, 0)] = DySample(num_ch_out, style="pl")
        elif self.upsample_mode == "dysample_lp":
            self.convs[("upconv_upsample", 2, 0)] = DySample(num_ch_out, style="lp")
        num_ch_in = num_ch_out + self.num_ch_enc[1]
        num_ch_out = self.num_ch_dec[2]

        self.convs[("upconv", 2, 1)] = C3k2(num_ch_in, num_ch_out)
        if self.upsample_mode == "dysample_pl":
            self.convs[("upconv_upsample", 1, 0)] = DySample(num_ch_out, style="pl")
        elif self.upsample_mode == "dysample_lp":
            self.convs[("upconv_upsample", 1, 0)] = DySample(num_ch_out, style="lp")
        num_ch_in = num_ch_out + self.num_ch_enc[0]
        num_ch_out = self.num_ch_dec[1]


        self.convs[("upconv", 1, 1)] = C3k2(num_ch_in, num_ch_out)

        self.convs[("dispconv", 1)] = Conv3x3(self.num_ch_dec[1],
                                              self.num_output_channels)

        self.convs[("dispconv", 2)] = Conv3x3(self.num_ch_dec[2],
                                              self.num_output_channels)
        self.convs[("dispconv", 3)] = Conv3x3(self.num_ch_dec[3],
                                              self.num_output_channels)

        self.convs[("dispconv", 4)] = Conv3x3(self.num_ch_dec[4],
                                              self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)
        self.outputs = {}

        self.scale_factor = 2.0

    def upsample(self, x: torch.Tensor, mode="nearest"):
        # 使用实例属性
        return F.interpolate(x, scale_factor=self.scale_factor, mode=mode)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        outputs = {}
        # decoder
        x = input_features[self.num_out_mix]  # input_features[-1]

        #x = [self.upsample(x, mode=self.upsample_mode)]
        x = [self.convs[("upconv_upsample", 4, 0)](x)]

        x += [input_features[3]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 4, 1)](x)

        f = self.convs[("dispconv", 4)](x)

        f = self.upsample(f, mode='bilinear')


        outputs[("disp", 3)] = self.sigmoid(f)
        #x = [upsample(x, mode=self.upsample_mode)]
        x = [self.convs[("upconv_upsample", 3, 0)](x)]
        x += [input_features[2]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 3, 1)](x)
        f = self.convs[("dispconv", 3)](x)
        f = self.upsample(f, mode='bilinear')

        outputs[("disp", 2)] = self.sigmoid(f)

        # x = [upsample(x, mode=self.upsample_mode)]
        x = [self.convs[("upconv_upsample", 2, 0)](x)]
        x += [input_features[1]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 2, 1)](x)
        f = self.convs[("dispconv", 2)](x)
        f = self.upsample(f, mode='bilinear')

        outputs[("disp", 1)] = self.sigmoid(f)

        #x = [upsample(x, mode=self.upsample_mode)]
        x = [self.convs[("upconv_upsample", 1, 0)](x)]
        x += [input_features[0]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 1, 1)](x)
        f = self.convs[("dispconv", 1)](x)
        f = self.upsample(f, mode='bilinear')

        outputs[("disp", 0)] = self.sigmoid(f)

        return outputs[("disp", 0)]




class FlexDepthDecoderLiteScale4DyPLC3KG8Export(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, scale="n", freeze=0,
                 opt=None):
        super(FlexDepthDecoderLiteScale4DyPLC3KG8Export, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips

        self.upsample_mode = "nearest"  # "bilinear"#'nearest'
        if opt is not None:
            if opt.upsample == "nearest":
                self.upsample_mode = "nearest"
            elif opt.upsample == "bilinear":
                self.upsample_mode = "bilinear"
            elif opt.upsample == "dysample_pl":
                self.upsample_mode = "dysample_pl"
            elif opt.upsample == "dysample_lp":
                self.upsample_mode = "dysample_lp"

        if self.upsample_mode in ["dysample_pl", "dysample_lp"]:
            self.use_all_upsample_learn = True
            self.use_dysample = True
        else:
            self.use_all_upsample_learn = False

        print("self.upsample_mode 【】", self.upsample_mode)
        self.scales = scales
        group = 8


        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([32, 64, 128, 256, 512])

        model_scales = {'n': [0.5, 0.25, 1024], 's': [0.5, 0.5, 1024], 'm': [0.5, 1.0, 512], 'l': [1.0, 1.0, 512],
                        'x': [1.0, 1.5, 512]}
        depth, width, max_channels = model_scales[scale]

        self.num_out_mix = 4  # max(self.scales)
        self.num_ch_dec = np.array([int(min(x * width, max_channels)) for x in self.num_ch_dec])
        self.num_ch_dec = [int(x) for x in self.num_ch_dec]
        print("num_ch_dec", self.num_ch_dec)

        self.freeze = freeze

        # decoder
        # self.convs = OrderedDict()
        self.convs = OrderedDict()
        num_ch_in = self.num_ch_enc[-1]

        self.convs[("upconv_upsample", 4, 0)] = DySample(num_ch_in, style="pl",groups=group)


        num_ch_out = self.num_ch_dec[4]
        num_ch_in += self.num_ch_enc[3]



        self.convs[("upconv", 4, 1)] = C3k2(num_ch_in, num_ch_out,c3k=True)

        self.convs[("upconv_upsample", 3, 0)] = DySample(num_ch_out, style="pl",groups=group)

        num_ch_in = num_ch_out + self.num_ch_enc[2]
        num_ch_out = self.num_ch_dec[3]



        self.convs[("upconv", 3, 1)] = C3k2(num_ch_in, num_ch_out,c3k=True)

        self.convs[("upconv_upsample", 2, 0)] = DySample(num_ch_out, style="pl",groups=8)
        num_ch_in = num_ch_out + self.num_ch_enc[1]
        num_ch_out = self.num_ch_dec[2]

        self.convs[("upconv", 2, 1)] = C3k2(num_ch_in, num_ch_out,c3k=True)

        self.convs[("upconv_upsample", 1, 0)] = DySample(num_ch_out, style="pl",groups=8)

        num_ch_in = num_ch_out + self.num_ch_enc[0]
        num_ch_out = self.num_ch_dec[1]



        self.convs[("upconv", 1, 1)] = C3k2(num_ch_in, num_ch_out,c3k=True)



        self.convs[("dispconv", 1)] = Conv3x3(self.num_ch_dec[1],
                                              self.num_output_channels)

        self.convs[(f"upsample_with_params_0", 1)] = DySample(self.num_ch_dec[1], style="pl", groups=group)



        self.convs[("dispconv", 2)] = Conv3x3(self.num_ch_dec[2],
                                              self.num_output_channels)
        self.convs[(f"upsample_with_params_0", 2)] = DySample(self.num_ch_dec[2], style="pl", groups=group)


        self.convs[("dispconv", 3)] = Conv3x3(self.num_ch_dec[3],
                                              self.num_output_channels)
        self.convs[(f"upsample_with_params_0", 3)] = DySample(self.num_ch_dec[3], style="pl", groups=group)



        self.convs[("dispconv", 4)] = Conv3x3(self.num_ch_dec[4],
                                              self.num_output_channels)
        self.convs[(f"upsample_with_params_0", 4)] = DySample(self.num_ch_dec[4], style="pl", groups=group)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)
        self.outputs = {}

        self.scale_factor = 2.0

    def upsample(self, x: torch.Tensor, mode="nearest"):
        # 使用实例属性
        return F.interpolate(x, scale_factor=self.scale_factor, mode=mode)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        outputs = {}
        # decoder
        x = input_features[self.num_out_mix]  # input_features[-1]

        #x = [self.upsample(x, mode=self.upsample_mode)]
        x = [self.convs[("upconv_upsample", 4, 0)](x)]

        x += [input_features[3]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 4, 1)](x)

        f = self.convs[(f"upsample_with_params_{0}", 4)](x)
        f = self.convs[("dispconv", 4)](f)

        # f = self.upsample(f, mode='bilinear')



        outputs[("disp", 3)] = self.sigmoid(f)
        #x = [upsample(x, mode=self.upsample_mode)]
        x = [self.convs[("upconv_upsample", 3, 0)](x)]
        x += [input_features[2]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 3, 1)](x)
        f = self.convs[(f"upsample_with_params_{0}", 3)](x)
        f = self.convs[("dispconv", 3)](f)
        #f = self.upsample(f, mode='bilinear')

        outputs[("disp", 2)] = self.sigmoid(f)

        # x = [upsample(x, mode=self.upsample_mode)]
        x = [self.convs[("upconv_upsample", 2, 0)](x)]
        x += [input_features[1]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 2, 1)](x)
        f = self.convs[(f"upsample_with_params_{0}", 2)](x)
        f = self.convs[("dispconv", 2)](f)
        # f = self.upsample(f, mode='bilinear')

        outputs[("disp", 1)] = self.sigmoid(f)

        #x = [upsample(x, mode=self.upsample_mode)]
        x = [self.convs[("upconv_upsample", 1, 0)](x)]
        x += [input_features[0]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 1, 1)](x)
        f = self.convs[(f"upsample_with_params_{0}", 1)](x)
        f = self.convs[("dispconv", 1)](f)
        # f = self.upsample(f, mode='bilinear')

        outputs[("disp", 0)] = self.sigmoid(f)

        return outputs[("disp", 0)]




class MuPredictor(nn.Module):
    """
    一个极其精简高效的网络，用于从两个特征图中预测标量 mu。
    mu 的范围被约束在 [0.01, 0.2] 之间。

    假设输入特征图F1, F2的形状为 [B, C, H, W] (PyTorch标准格式)
    """

    def __init__(self, in_channels,target_mu=0.1,use_diff=False):
        """
        in_channels: 单个特征图的通道数 (例如 64, 128, ...)
        """
        super().__init__()
        self.MU_LOWER_BOUND = 0.08
        self.MU_RANGE = 0.07  # 0.2 - 0.01
        self.use_diff = use_diff
        if use_diff:
            print("Using feature difference to predict mu")
        # 1. 特征提取和空间压缩 (已简化)
        # 我们使用两个带步长的卷积来快速压缩空间信息
        if use_diff:
            self.encoder = nn.Sequential(
                # Conv 1: [B, 2*C, H, W] -> [B, 32, H/2, W/2]
                nn.Conv2d(in_channels , 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),

                # Conv 2: [B, 32, H/2, W/2] -> [B, 16, H/4, W/4]
                nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )
        else:
            self.encoder = nn.Sequential(
                # Conv 1: [B, 2*C, H, W] -> [B, 32, H/2, W/2]
                nn.Conv2d(in_channels * 2, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),

                # Conv 2: [B, 32, H/2, W/2] -> [B, 16, H/4, W/4]
                nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )

        # 2. 全局平均池化
        # [B, 16, H/4, W/4] -> [B, 16, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 3. 回归头部 (MLP) - 简化为单个全连接层
        # [B, 16] -> [B, 1] (1个无约束的 logit for mu)
        self.fc_head = nn.Linear(16, 1)  # 直接从16维特征预测1个值

        self._initialize_weights(target_mu)

    def _calculate_bias(self, target_mu):
        """
        根据目标 mu 和约束范围 [0.01, 0.2] 自动计算偏置值。
        """
        # 检查 target_mu 是否在有效范围内
        if not (self.MU_LOWER_BOUND < target_mu < (self.MU_LOWER_BOUND + self.MU_RANGE)):
            print(f"Warning: target_mu={target_mu} is outside the strict range (0.01, 0.2). Using 0.105 instead.")
            target_mu = 0.105  # 如果超出范围，使用默认中心值 0.105

        # 1. 计算 sigmoid 的目标概率 p
        # p = (target_mu - lower_bound) / range_width
        p = (target_mu - self.MU_LOWER_BOUND) / self.MU_RANGE

        # 2. 计算 logit (即 bias)
        # logit = log(p / (1 - p))
        # 使用 torch.log() 或 math.log()。此处使用 math.log()
        bias_value = math.log(p / (1 - p))

        return bias_value

    def _initialize_weights(self, target_mu):
        """
        初始化网络权重，并计算全连接层的偏置项。
        """
        bias_value = self._calculate_bias(target_mu)
        print(f"【Target mu={target_mu}, Calculated initial bias for FC layer: {bias_value:.4f}】")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 权重初始化
                init.kaiming_uniform_(m.weight, nonlinearity='relu')

                # 偏置项初始化
                init.constant_(m.bias, bias_value)


    def forward(self, f1, f2):
        """
        f1, f2: 形状为 [B, C, H, W] 的输入特征图
        """
        # 1. 合并输入: [B, C, H, W] + [B, C, H, W] -> [B, 2*C, H, W]
        # print(f2.shape)
        if self.use_diff:
            x = f1 - f2
        else:
            x = torch.cat([f1, f2], dim=1)

        # 2. 编码和压缩
        x = self.encoder(x)

        # 3. 全局池化
        x = self.global_pool(x)  # [B, 16, 1, 1]
        x = torch.flatten(x, 1)  # [B, 16]

        # 4. 回归 logit_mu
        # logit_mu 的形状为 [B, 1]
        logit_mu = self.fc_head(x)

        # 5. 范围约束 (核心)
        # torch.sigmoid() 将输出映射到 (0, 1)
        # (sigmoid * 范围宽度) + 范围下限

        # mu: 范围 [0.01, 0.2], 宽度 = 0.19
        # 0.08 宽度 0.08
        mu = (torch.sigmoid(logit_mu) * self.MU_RANGE) + self.MU_LOWER_BOUND

        # 为了方便广播，我们返回 [B, 1, 1, 1] 形状
        # 这样它可以直接与 [B, 1, H, W] 的 disp_f 进行计算
        return mu.unsqueeze(-1).unsqueeze(-1)