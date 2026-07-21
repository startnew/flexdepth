#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/22 16:05
# @Author  : zhuzhaowen
# @File    : yolo11.py
# @Software: PyCharm
# @desc    : "Initialize model from config file"

import sys

sys.path.append("../")
import logging as LOGGER
import torch
import torch.nn as nn
import math
from pathlib import Path
from copy import deepcopy

from utils_add.conv import Conv, Concat
from utils_add.block import C2PSA, C3k2, SPPF
from utils_add.torch_utils import fuse_conv_and_bn, model_info

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

config_file = "./cfg/models/yolo11-dep.yaml"


def make_divisible(x, divisor):
    """
    Returns the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int | torch.Tensor): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def parse_model(model_dict, ch, scale="n", verbose=True,all_model=False):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (model_dict.get(x) for x in ("nc", "activation", "scales"))
    print("【scales】",scales)
    depth, width, kpt_shape = (model_dict.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:

        # scale = model_dict.get("scale")
        scale = scale
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    if all_model:
        need_opt = model_dict["backbone"] + model_dict["head"]
    else:
        need_opt = model_dict["backbone"]
    for i, (f, n, m, args) in enumerate(need_opt):  # from, number, module, args
        # print(depth, width, max_channels,"scale",scale)
        # print("(f, n, m, args)", (f, n, m, args))

        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module

        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                except ValueError:
                    pass
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        # print(n * depth,"n * depth",n,depth,width,max_channels)
        # Conv C3k2 SPPF C2PSA
        if m in {

            Conv,

            SPPF,

            C2PSA,

            C3k2,

        }:
            # print("args", args)
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]

            if m in {

                C3k2,

                C2PSA,
            }:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # elif m in {MonoDepth}:
        #     args.append([ch[x] for x in f])
        #     if m is MonoDepth:
        #         args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        #     if m in {MonoDepth}:
        #         m.legacy = legacy
        else:
            c2 = ch[f]
        # print("m args", args,m)
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def colorstr(*input):
    r"""
    Colors a string based on the provided color and style arguments. Utilizes ANSI escape codes.
    See https://en.wikipedia.org/wiki/ANSI_escape_code for more details.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str | Path): A sequence of strings where the first n-1 strings are color and style arguments,
                      and the last string is the one to be colored.

    Supported Colors and Styles:
        Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        Misc: 'end', 'bold', 'underline'

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr("blue", "bold", "hello world")
        >>> "\033[34m\033[1mhello world\033[0m"
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


import pickle


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """Attempt to find a class, returning SafeClass if not among safe modules."""
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            "torch"

            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            print("module", module)
            return SafeClass


class YOLO11Encoder():
    def __init__(self, cfg, ch=3, nc=None, scale="n", use_last=False,all_model=False,feature_idxs=[],feature_chanel_idxs=[]):
        import yaml  # for torch hub
        self.yaml_file = Path(cfg).name
        with open(cfg, encoding='ascii', errors='ignore') as f:
            self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ori_ch = ch
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if ori_ch != ch:
            print("Using multi-image channels:", ori_ch)
            ch = ori_ch
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, scale=scale,all_model=all_model)  # model, savelist
        #print(self.model)

        if "cls" in cfg:
            if use_last:
                self.feature_chanel_idxs = [0, 1, 3, 5, 7]
                self.feature_idxs = [0, 2, 4, 6, 9]
            else:
                self.feature_chanel_idxs = [0, 1, 3, 5, 7]
                self.feature_idxs = [0, 1, 3, 5, 7]
                self.feature_idxs = [0, 1, 3, 5, 10]


        else:
            if use_last:
                self.feature_chanel_idxs = [0, 1, 3, 5, 7]
                self.feature_idxs = [0, 2, 4, 6, 10]
                #self.feature_idxs = [0, 2, 4, 6, 10]
            else:
                self.feature_chanel_idxs = [0, 1, 3, 5, 7]
                self.feature_idxs = [0, 1, 3, 5, 7]
                self.feature_idxs = [0, 1, 3, 5, 10]
        if len(feature_idxs) > 0:
            self.feature_idxs = feature_idxs
            self.feature_chanel_idxs = feature_chanel_idxs
        #self.feature_idxs = [0, 1, 3, 5, 10]
        # Use fuse during testing
        # self.fuse()
        self.model = self.model[:max(self.feature_idxs) + 1]
        self.info()

        print("Feature extraction layers:", self.feature_idxs)

    def fuse(self, verbose=False):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv,)) and hasattr(m, "bn"):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(
            isinstance(v, bn) for v in self.model.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=False, imgsz=[640, 192]):
        """
        Prints model information.

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self.model, detailed=detailed, verbose=verbose, img_size=imgsz)





if __name__ == "__main__":
    model = YOLO11Encoder(cfg="../cfg/models/yolo11-dep-encoder.yaml", ch=3)
    print(model.model)
    print(model)
