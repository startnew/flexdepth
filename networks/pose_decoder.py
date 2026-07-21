from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


class PoseDecoder(nn.Module):
    """
       将编码器输出的特征解码为相机位姿 (轴角旋转和平移向量) 的模块
       """
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        """
               初始化 PoseDecoder 模块。

               参数:
                   num_ch_enc (list): 编码器各层输出的通道数列表。通常取最后一个元素的通道数。
                   num_input_features (int): 输入的图像帧数。
                   num_frames_to_predict_for (int, optional): 需要预测位姿的帧数。默认是输入帧数减一。
                                                            例如，如果输入 3 帧 (参考帧 + 2个其他帧)，通常预测 2 个位姿 (参考帧到其他帧)。
                   stride (int, optional): 卷积层的步长。默认为 1。
               """
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc

        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for


        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
        # 对输出特征图进行全局平均池化 (spatial global average pooling)
        # 计算高度维度 (维度 2) 和宽度维度 (维度 3) 的平均值，将空间信息聚合成一个向量
        out = out.mean(3).mean(2)

        # 将池化后的输出重塑形状，并乘以一个小的缩放因子 0.01
        # 重塑后的形状为 (batch_size, num_frames_to_predict_for, 1, 6)，
        # 最后一个维度包含 6 个位姿参数 (3个轴角, 3个平移)
        # 乘以 0.01 是常见的做法，有助于在训练初期保持位姿值较小

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)
        #  从输出中提取轴角向量 (前 3 个参数)
        axisangle = out[..., :3]
        # 从输出中提取平移向量 (后 3 个参数)
        translation = out[..., 3:]
        # import random
        # if random.random() < 0.001:
        #     print("预测的 axisangle",axisangle.shape,axisangle[0])
        #     print("预测的 translation", translation.shape,translation[0])

        return axisangle, translation
