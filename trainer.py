from __future__ import absolute_import, division, print_function
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402

import random



import time
import wandb
import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F

from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
try:
    from torch import amp
except Exception as e:
    print(e,"amp")
from utils_add.torch_utils import profile_once


from tqdm import tqdm
import math

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils_add.torch_utils import  select_device, torch_distributed_zero_first,  \
    de_parallel, one_cycle
from utils_add.general import init_seeds






from pyclbr import Function
from typing import Sequence
import torch

from prefetch_generator import BackgroundGenerator

try:
    import ray.train as ray_train
    import ray
except Exception as e:
    ray = None


class bcolors:
    HEADER = "\033[95m{0}\033[0m"
    OKBULE = '\033[94m{0}\033[0m'
    OKGREEN = '\033[32m{0}\033[0m'
    WERNING = '\033[93m{0}\033[0m'
    FAIL = '\033[31m{0}\033[0m'

class DataLoaderX(DataLoader):
    # max_prefetch = 6
    max_prefetch = 12

    print("Prefetch queue length (reduce if memory is insufficient):", max_prefetch)
    def __iter__(self):
        max_prefetch = DataLoaderX.max_prefetch
        return BackgroundGenerator(super().__iter__(),max_prefetch=max_prefetch)


class DataPrefetcher():
    def __init__(self, loader, opt):
        print("Using DataPrefetcher with CUDA stream for GPU tensor copy")
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.preload()

    def re_load(self):
        self.loader = iter(self.ori_loader)
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k != 'meta':
                    self.batch[k] = self.batch[k].to(device=self.opt.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch


from contextlib import contextmanager



class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        self.models = {}
        if self.opt.use_step_2:
            self.models_f = {}
        self.parameters_to_train = []
        self.rank = self.opt.global_rank
        use_iter = False
        self.use_iter = use_iter
        print("self.rank", self.rank)
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        opt = self.opt
        device = select_device(opt.device, batch_size=opt.batch_size, local_rank=self.opt.local_rank)
        self.device = device
        print("Current device:", self.device, self.rank)

        init_seeds(seed=1 + self.rank, deterministic=False)

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # Initialize encoder
        self.init_encoder(step1=True)
        # Initialize depth decoder
        self.init_depth(step1=True)
        # Initialize pose network
        self.init_pose_net()
        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            print("[predictive_mask] needs separate decoder info")
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            # self.parameters_to_train += list(self.models["predictive_mask"].parameters())
        #  freeze = []  # parameter names to freeze (full or partial)
        # Initialize optimizer
        self.init_optim_data()

        cuda = self.device.type != 'cpu'
        self.cuda = cuda
        # Initialize learning rate scheduler
        self.init_scheduler()

        self.init_ddp_syncbn()
        self.init_dataset()
        # todo
        # self.ema = ModelEMA(self.models) if self.rank in [-1, 0] else None
        # print(self.ema)

        if self.opt.load_weights_folder is not None:
            if self.opt.dy_mu and "MuPredictor" not in self.opt.models_to_load:
                self.opt.models_to_load.append("MuPredictor")
            self.load_model()
        if self.opt.resume:
            last_weights_dir_p  = os.path.join(self.log_path, "models", )
            if os.path.exists(last_weights_dir_p):
                last_dirs = [int(x.split("_")[1]) for x in os.listdir(last_weights_dir_p) if "weights" in x]
                if len(last_dirs) > 0:

                    last_dirs.sort()
                    last_num = last_dirs[-1]
                    if last_num == self.opt.num_epochs -1 and not self.opt.use_step_2:
                        print("Training already completed, exiting early")
                        exit()
                    elif last_num == self.opt.num_epochs + self.opt.step_2_epoch - 1 and self.opt.use_step_2:
                        print("Training already completed, exiting early (STEP2)")
                        exit()

                    if last_num > self.opt.num_epochs:
                        last_num = self.opt.num_epochs -1
                    else:
                        pass
                    self.opt.load_weights_folder= os.path.join(self.log_path, "models", "weights_{}".format(last_num))
                    print("Resuming from last weights:", self.opt.load_weights_folder)
                    self.opt.start_epoch = last_num + 1

                    self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        self.writers = {}
        if self.rank in [-1, 0]:
            for mode in ["train", "val"]:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)
            if self.opt.use_gmsd:
                from layers import FSIM,GMSD
                #self.fsim = FSIM()
                #self.fsim.to(self.device)
                self.gmsd =GMSD()
                self.gmsd.to(self.device)


        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("use opt:", self.opt)
        if self.rank in [0, -1]:
            self.save_opts()

    def init_dataset(self):
        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        print("[Dataset]", self.opt.dataset, self.opt.split)
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        if self.opt.use_step_2 and not self.opt.skip_step1:
            self.num_total_steps = num_train_samples // self.opt.batch_size * (self.opt.num_epochs + self.opt.step_2_epoch)
        elif self.opt.use_step_2 and self.opt.skip_step1:
            self.num_total_steps = num_train_samples // self.opt.batch_size * (
                        self.opt.step_2_epoch)
        else:
            self.num_total_steps = num_train_samples // self.opt.batch_size * (
                        self.opt.num_epochs )

        num_workers = min([os.cpu_count() // self.opt.world_size, self.opt.batch_size if self.opt.batch_size > 1 else 0,
                           self.opt.num_workers])  # number of workers
        print("[num_workers]:", num_workers, "batch_size", self.opt.batch_size, "self.opt.height, self.opt.width",
              self.opt.height, self.opt.width)

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        shuffle = True
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=shuffle,
                                                                  num_replicas=dist.get_world_size(),
                                                                  rank=self.opt.local_rank) if self.rank != -1 else None
        self.train_dataset = train_dataset
        use_prefetch_generator = False
        use_dataprefetcher = False


        if self.rank != -1:
            use_prefetch_generator = False
            use_dataprefetcher = False

        self.use_dataprefetcher = use_dataprefetcher

        if sampler is not None:

            # DataLoaderX
            if use_prefetch_generator:
                self.train_loader = DataLoaderX(
                    train_dataset, self.opt.batch_size, shuffle=shuffle and sampler is None,
                    num_workers=num_workers, pin_memory=True, drop_last=True, sampler=sampler)
            else:
                self.train_loader = DataLoader(
                    train_dataset, self.opt.batch_size, shuffle=shuffle and sampler is None,
                    num_workers=num_workers, pin_memory=True, drop_last=True, sampler=sampler)
        else:
            #
            if use_prefetch_generator:
                self.train_loader = DataLoaderX(
                    train_dataset, self.opt.batch_size, shuffle=shuffle and sampler is None,
                    num_workers=num_workers, pin_memory=True, drop_last=True
                )
            else:
                self.train_loader = DataLoader(
                    train_dataset, self.opt.batch_size, shuffle=shuffle and sampler is None,
                    num_workers=num_workers, pin_memory=True, drop_last=True
                )

        # self.train_iter = iter(self.train_loader)
        if self.rank in [-1, 0]:
            val_dataset = self.dataset(self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
                                       self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
            self.val_loader = DataLoader(
                val_dataset, self.opt.batch_size, True,
                num_workers=num_workers, pin_memory=True, drop_last=True)
            self.val_iter = iter(self.val_loader)

        print("Using split:\n  ", self.opt.split)
        if self.rank in [-1, 0]:
            print("There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset), len(val_dataset)))
        else:
            print("There are {:d} training items \n".format(
                len(train_dataset)))

    def init_ddp_syncbn(self):
        if self.opt.local_rank != -1:
            assert torch.cuda.device_count() > self.opt.local_rank
            torch.cuda.set_device(self.opt.local_rank)
            from datetime import  timedelta

            os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
            dist.init_process_group(backend='nccl', rank=self.opt.local_rank, world_size=self.opt.world_size,
                                    timeout=timedelta(seconds=3600), )  # distributed backend
            print("init_process_group ok")
            assert self.opt.batch_size % self.opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
            self.opt.batch_size = self.opt.total_batch_size // self.opt.world_size

        if self.cuda and self.rank != -1 and torch.cuda.device_count() > 1:
            useDP = False
            if useDP:
                new_models = {}
                for k, v in self.models.items():
                    new_models[k] = torch.nn.DataParallel(self.models[k])

                del self.models
                self.models = new_models  # torch.nn.DataParallel(self.models)
            print("use multi gpu", )
        if self.opt.sync_bn and self.cuda and self.rank != -1:
            new_models = {}
            for k, v in self.models.items():
                new_models[k] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models[k]).to(self.device)
            del self.models
            self.models = new_models
            print('Using SyncBatchNorm()')

        # DDP mode
        if self.cuda and self.rank != -1:
            new_models = {}

            for k, v in self.models.items():
                # broadcast_buffers=False, fixes [rank0]: RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
                # https://blog.csdn.net/qq_39237205/article/details/125728708
                new_models[k] = DDP(self.models[k], device_ids=[self.opt.local_rank], output_device=self.opt.local_rank,
                                    broadcast_buffers=False,
                                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in
                                                               self.models[k].modules()))
                # any(isinstance(layer, nn.MultiheadAttention) for layer in self.models[k].modules())
                # del self.models[k]
            self.models = new_models
        if self.opt.amp:
            print("amp checks passed")
            self.scaler = amp.GradScaler(enabled=self.cuda, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5,
                                         growth_interval=2000, )

    def init_scheduler(self):
        if self.opt.scheduler == "cos":
            self.lf = one_cycle(1, self.opt.lrf, self.opt.num_epochs)  # cosine 1->hyp['lrf']
            self.model_lr_scheduler = optim.lr_scheduler.LambdaLR(self.model_optimizer, lr_lambda=self.lf)
        elif self.opt.scheduler == "linear":
            self.lf = lambda x: max(1 - x / self.opt.num_epochs, 0) * (1.0 - self.opt.lrf) + self.opt.lrf  # linear

            self.model_lr_scheduler = optim.lr_scheduler.LambdaLR(self.model_optimizer, lr_lambda=self.lf)
        elif self.opt.scheduler == "step":
            self.lf = lambda epoch: 0.1 ** (epoch // self.opt.scheduler_step_size)
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(
                self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        elif self.opt.scheduler == "ExponentialLR":
            # lf only takes effect during warm-up epochs, so only restore LR changes for those epochs
            self.lf = lambda epoch: 0.9 ** epoch if epoch != self.opt.num_epochs - 1 else self.opt.learning_rate
            self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(
                self.model_optimizer, 0.9)

    def init_optim_data(self):
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()

        g = [], [], []  # optimizer parameter groups
        g_encoder = []
        use_encoder_half = False
        for k, model in self.models.items():
            for module_name, module in model.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    fullname = f"{module_name}.{param_name}" if module_name else param_name
                    if k in ["encoder"] and use_encoder_half:
                        g_encoder.append(param)
                    elif "bias" in fullname:  # bias (no decay)
                        g[2].append(param)
                    elif isinstance(module, bn):  # weight (no decay)
                        g[1].append(param)
                    else:
                        g[0].append(param)

        self.accumulate = max(round(self.opt.nbs / self.opt.batch_size), 1)  # accumulate loss before optimizing
        use_group = False

        self.parameters_to_train = g[0] + g[1] + g[2]

        print("[use group]", use_group, len(self.parameters_to_train))
        if use_group:
            params_ = g[2]
        else:
            params_ = self.parameters_to_train
        weight_decay = self.opt.weight_decay * self.opt.batch_size * self.accumulate / self.opt.nbs  # scale weight_decay
        if self.opt.optim in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            if use_group:
                optimizer = getattr(optim, self.opt.optim, optim.Adam)(params_, lr=self.opt.learning_rate,
                                                                       betas=(self.opt.momentum, 0.999),
                                                                       weight_decay=0.0)
            else:
                optimizer = getattr(optim, self.opt.optim, optim.Adam)(params_, lr=self.opt.learning_rate,
                                                                       )
        elif self.opt.optim in {"RMSProp", }:
            if use_group:
                optimizer = optim.RMSprop(params_, lr=self.opt.learning_rate, momentum=self.opt.momentum)
            else:
                optimizer = optim.RMSprop(params_, lr=self.opt.learning_rate, )
        elif self.opt.optim in {"SGD", }:
            if use_group:
                optimizer = optim.SGD(params_, lr=self.opt.learning_rate, momentum=self.opt.momentum, nesterov=True)
            else:
                optimizer = optim.SGD(params_, lr=self.opt.learning_rate, )

        self.model_optimizer = optimizer

        if use_encoder_half:
            print("[Halving encoder learning rate]")
            self.model_optimizer.add_param_group(
                {"params": g_encoder, "weight_decay": weight_decay, "lr": self.opt.learning_rate / 2})

        if use_group:
            if use_encoder_half:
                self.model_optimizer.add_param_group(
                    {"params": g[0], "weight_decay": weight_decay})  # add g0 with weight_decay
                self.model_optimizer.add_param_group(
                    {"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
            else:
                self.model_optimizer.add_param_group(
                    {"params": g[0], "weight_decay": weight_decay})  # add g0 with weight_decay
                self.model_optimizer.add_param_group(
                    {"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)

            print(
                f"('optimizer:') {type(self.model_optimizer).__name__}(lr={self.opt.learning_rate}, momentum={self.opt.momentum}) with parameter groups "
                f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={weight_decay}), {len(g[2])} bias(decay=0.0)'
            )
        print(self.model_optimizer, "self.model_optimizer")

    def init_encoder(self,step1=False):
        if self.opt.encoder_model_type == "resnet":
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            if self.opt.use_step_2 and not step1:
                self.models_f["encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers, self.opt.weights_init == "pretrained")
                self.models_f["encoder_0"] = networks.ResnetEncoder(
                    self.opt.num_layers, self.opt.weights_init == "pretrained")
        elif "yolo11" in self.opt.encoder_model_type:
            print("model scale:", self.opt.encoder_model_type.replace("yolo11", ""))
            self.models["encoder"] = networks.YOLOEncoder(
                self.opt.weights_init == "pretrained", scale=self.opt.encoder_model_type.replace("yolo11", ""))
            if self.opt.use_step_2 and not step1:
                self.models_f["encoder"] = networks.YOLOEncoder(
                    self.opt.weights_init == "pretrained", scale=self.opt.encoder_model_type.replace("yolo11", ""))
                self.models_f["encoder_0"] = networks.YOLOEncoder(
                    self.opt.weights_init == "pretrained", scale=self.opt.encoder_model_type.replace("yolo11", ""))
        self.models["encoder"].to(self.device)
        if self.opt.freeze_encoder:
            print("Freezing encoder, only training decoder")
            for param in self.models["encoder"].parameters():
                param.requires_grad = False

        if self.opt.use_step_2 and not step1:
            for param in self.models_f["encoder_0"].parameters():
                param.requires_grad = False
            for param in self.models_f["encoder"].parameters():
                param.requires_grad = False
            self.models_f["encoder_0"].to(self.device)
            self.models_f["encoder"].to(self.device)


    def init_depth(self,step1=False):

        print("Initializing depth decoder network")

        if self.opt.decoder_model_type == "ori":
            self.models["depth"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales, freeze=0)
            if self.opt.use_step_2 and not step1:
                self.models_f["depth"] = networks.DepthDecoder(
                    self.models["encoder"].num_ch_enc, self.opt.scales, freeze=1)
                self.models_f["depth_0"] = networks.DepthDecoder(
                    self.models["encoder"].num_ch_enc, self.opt.scales, freeze=2)

        elif "flex" in self.opt.decoder_model_type:
            self.models["depth"] = networks.FlexDepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                scale=self.opt.decoder_model_type.replace("flex", ""),opt=self.opt)
            if self.opt.use_step_2 and not step1:
                self.models_f["depth"] = networks.FlexDepthDecoder(
                    self.models["encoder"].num_ch_enc, self.opt.scales,
                    scale=self.opt.decoder_model_type.replace("flex", ""),
                    freeze=1,opt=self.opt)
                self.models_f["depth_0"] = networks.FlexDepthDecoder(
                    self.models["encoder"].num_ch_enc, self.opt.scales,
                    scale=self.opt.decoder_model_type.replace("flex", ""),
                    freeze=2,opt=self.opt)
        self.models["depth"].to(self.device)
        if self.opt.use_step_2 and not step1:
            for param in self.models_f["depth_0"].parameters():
                param.requires_grad = False
            for param in self.models_f["depth"].parameters():
                param.requires_grad = False
            self.models_f["depth"].to(self.device)
            self.models_f["depth_0"].to(self.device)

    def init_pose_net(self):
        pre_train = True#self.opt.weights_init == "pretrained"
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    pre_train,
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)
                if self.opt.use_step_2:
                    self.models_f["pose_encoder"] = networks.ResnetEncoder(
                        self.opt.num_layers,
                        pre_train,
                        num_input_images=self.num_pose_frames)
                    self.models_f["pose"] = networks.PoseDecoder(
                        self.models["pose_encoder"].num_ch_enc,
                        num_input_features=1,
                        num_frames_to_predict_for=2)
            elif self.opt.pose_model_type == "double_resnet":
                self.models["pose_encoder"] = networks.DoubleResnetEncoder(
                    self.opt.num_layers,
                    pre_train,
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                # self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

                if self.opt.use_step_2:
                    self.models_f["pose_encoder"] = networks.DoubleResnetEncoder(
                        self.opt.num_layers,
                        pre_train,
                        num_input_images=self.num_pose_frames)
                    self.models_f["pose"] = networks.PoseDecoder(
                        self.models["pose_encoder"].num_ch_enc,
                        num_input_features=1,
                        num_frames_to_predict_for=2)



            elif self.opt.pose_model_type == "double_resnet_same":
                self.models["pose_encoder"] = networks.DoubleResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames, same_param=True)

                self.models["pose_encoder"].to(self.device)
                # self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)
                if self.opt.use_step_2:
                    self.models_f["pose_encoder"] = networks.DoubleResnetEncoder(
                        self.opt.num_layers,
                        self.opt.weights_init == "pretrained",
                        num_input_images=self.num_pose_frames, same_param=True)
                    self.models_f["pose"] = networks.PoseDecoder(
                        self.models["pose_encoder"].num_ch_enc,
                        num_input_features=1,
                        num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            if self.opt.use_step_2:
                for param in self.models_f["pose"].parameters():
                    param.requires_grad = False
                for param in self.models_f["pose_encoder"].parameters():
                    param.requires_grad = False
                self.models_f["pose"].to(self.device)
                self.models_f["pose_encoder"].to(self.device)

    def set_step2(self):
        # Initialize step2 modules

        # Initialize encoder
        del self.models["encoder"]
        self.init_encoder()
        # Initialize depth decoder
        del self.models["depth"]
        self.init_depth()
        self.init_pose_net()
        # Initialize pose network

        save_folder_last = os.path.join(self.log_path, "models", "weights_{}".format(self.opt.num_epochs - 1))
        if self.opt.dataset == "cityscapes_preprocessed":
            first_epoch_num = 1
        else:
            first_epoch_num = 2
        save_folder_first = os.path.join(self.log_path, "models", "weights_{}".format(first_epoch_num))  # 0
        print(save_folder_last, save_folder_first, "save_folder_last,save_folder_first")
        encoder_dict_last = torch.load(
            f"{save_folder_last}/encoder.pth")
        encoder_dict_first = torch.load(
            f"{save_folder_first}/encoder.pth")

        depth_dict_last = torch.load(
            f"{save_folder_last}/depth.pth")
        depth_dict_first = torch.load(
            f"{save_folder_first}/depth.pth")

        pose_encoder_last = torch.load(
            f"{save_folder_last}/pose_encoder.pth")

        pose_last = torch.load(
            f"{save_folder_last}/pose.pth")

        model_dict = self.models_f["encoder_0"].state_dict()

        self.models_f["encoder_0"].load_state_dict({k: v for k, v in encoder_dict_first.items() if k in model_dict})

        model_dict = self.models_f["encoder"].state_dict()

        self.models_f["encoder"].load_state_dict({k: v for k, v in encoder_dict_last.items() if k in model_dict})

        self.models_f["depth_0"].load_state_dict({k: v for k, v in depth_dict_first.items() if k in self.models_f["depth_0"].state_dict()})  # ({k: v for k, v in depth_dict_first.items() if k in model_dict})


        self.models_f["depth"].load_state_dict(
            {k: v for k, v in depth_dict_last.items() if k in self.models_f["depth"].state_dict()})  # {k: v for k, v in depth_dict_last.items() if k in model_dict})

        self.models_f["pose_encoder"].load_state_dict(
            pose_encoder_last)

        self.models_f["pose"].load_state_dict(
            pose_last)

        self.models_f["encoder"].to(self.device)
        self.models_f["encoder_0"].to(self.device)

        self.models_f["depth_0"].to(self.device)
        self.models_f["depth"].to(self.device)
        self.models_f["pose_encoder"].to(self.device)
        self.models_f["pose"].to(self.device)

        # Re-declare frozen parameters as safety measure
        for param in self.models_f["encoder"].parameters():
            param.requires_grad = False

        for param in self.models_f["encoder_0"].parameters():
            param.requires_grad = False

        for param in self.models_f["depth"].parameters():
            param.requires_grad = False

        for param in self.models_f["depth_0"].parameters():
            param.requires_grad = False

        for param in self.models_f["pose_encoder"].parameters():
            param.requires_grad = False

        for param in self.models_f["pose"].parameters():
            param.requires_grad = False

        self.parameters_to_train = []
        self.parameters_to_train += list(self.models["depth"].parameters())
        # step2: re-enable automasking
        self.opt.disable_automasking = False

        del self.models["pose_encoder"]
        del self.models["pose"]

        if self.opt.use_var_net:
            print("[Initializing var net]")

            self.models["var_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=3)
            self.models["var_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["var_encoder"].parameters())

            self.models["var"] = networks.VarDecoder(self.models_f["pose_encoder"].num_ch_enc, self.opt.scales)
            self.models["var"].to(self.device)
            if self.opt.dy_mu:

                if self.opt.dataset == "cityscapes_preprocessed":
                    target_mu = 0.12
                else:
                    target_mu = 0.1
                if  self.opt.diff_dy_mu:
                    use_diff = True
                else:
                    use_diff = False
                self.models["MuPredictor"] = networks.MuPredictor(self.models_f["depth"].num_ch_dec[1],target_mu=target_mu,use_diff=use_diff)
                self.models["MuPredictor"].to(self.device)

                if self.opt.ada_dy_mu:
                    print("MuPredictor use signle learn")
                    mu_params = list(self.models["MuPredictor"].parameters())
                else:
                    self.parameters_to_train += list(self.models["MuPredictor"].parameters())
                    pass

            self.parameters_to_train += list(self.models["var"].parameters())
        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())
        if self.opt.same_lr:
            self.params = [{
                "params": self.parameters_to_train,
                "lr": self.opt.learning_rate

            },
                {
                    "params": list(self.models["encoder"].parameters()),
                    "lr": self.opt.learning_rate

                }]
        else:
            if self.opt.ada_dy_mu:
                print("MU using separate learning rate:", self.opt.mu_learning_rate)
                self.params = [{
                    "params": self.parameters_to_train,
                    "lr": 1e-4

                },
                    {
                        "params": list(self.models["encoder"].parameters()),
                        "lr": self.opt.learning_rate

                    },
                    {
                        "params": mu_params,
                        "lr": self.opt.mu_learning_rate

                    }
                ]

            else:


                self.params = [{
                    "params": self.parameters_to_train,
                    "lr": 1e-4

                },
                    {
                        "params": list(self.models["encoder"].parameters()),
                        "lr": self.opt.learning_rate

                    }]
        del self.model_lr_scheduler
        del self.model_optimizer
        # self.model_optimizer = optim.AdamW(self.params)
        use_group = False
        params_ = self.params
        if self.opt.optim in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            if use_group:
                optimizer = getattr(optim, self.opt.optim, optim.Adam)(params_)
            else:
                optimizer = getattr(optim, self.opt.optim, optim.Adam)(params_
                                                                       )
        elif self.opt.optim in {"RMSProp", }:
            if use_group:
                optimizer = optim.RMSprop(params_,  momentum=self.opt.momentum)
            else:
                optimizer = optim.RMSprop(params_,  )
        elif self.opt.optim in {"SGD", }:
            if use_group:
                optimizer = optim.SGD(params_, momentum=self.opt.momentum, nesterov=True)
            else:
                optimizer = optim.SGD(params_ )

        self.model_optimizer = optimizer
        self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.model_optimizer, 0.8)  # 0.8
        if self.opt.step2_scheduler == "ExponentialLR_08":
            self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(
                self.model_optimizer, 0.8)  # 0.8
        elif self.opt.step2_scheduler == "ExponentialLR_09":
            self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(
                self.model_optimizer, 0.9)  # 0.8
        else:
            self.init_scheduler()

        if self.opt.resume:
            last_weights_dir_p  = os.path.join(self.log_path, "models",)
            if os.path.exists(last_weights_dir_p):

                last_dirs = [int(x.split("_")[1]) for x  in os.listdir(last_weights_dir_p) if "weights" in x]
                last_dirs.sort()
                if len(last_dirs) > 0:
                    last_num = last_dirs[-1]

                    if last_num > self.opt.num_epochs:
                        self.opt.load_weights_folder= os.path.join(self.log_path, "models", "weights_{}".format(last_num))
                        print("Resuming from last STEP2 weights:", self.opt.load_weights_folder)
                        self.load_model(step2=True)
                        self.opt.start_epoch = last_num + 1


    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self, tune=False):
        """Run the entire training pipeline
        """
        self.use_ray_tune = False
        if tune:
            print("Using ray_train")
            self.use_ray_tune = True
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        if self.opt.start_epoch > 0:
            self.get_epoch_iters()
            self.step = self.iters_every_epoch * self.opt.start_epoch
            print(f"train resume from epoch {self.opt.start_epoch} step:{self.step}")
            for i in range(self.opt.start_epoch - 1):
                self.model_lr_scheduler.step()

        if self.opt.all_mem:
            print("Loading all training data into memory")
            from pympler import asizeof
            st = time.perf_counter()
            self.data = []
            for i, dat in tqdm(enumerate(self.train_loader)):
                self.data.append(dat)

            ed = time.perf_counter()

            # Use asizeof.asizeof to get total memory size in bytes
            total_size = asizeof.asizeof(self.data)

            # Convert bytes to MB
            total_size_mb = total_size / (1024 * 1024)

            print(f"The object occupies {total_size_mb:.2f} MB")
            print("All training data loaded into memory, time:", ed - st, len(self.data))


        else:
            self.data = self.train_loader
            print("Training data batch count:", len(self.data))

        num_batch = len(self.data)
        print(
            f"Training data: {num_batch} batches, batch_size={self.opt.batch_size}, total_batch_size={self.opt.total_batch_size}")
        self.num_warm_steps = max(round(self.opt.warmup_epochs * num_batch),
                                  100) if self.opt.warmup_epochs > 0 else -1  # warmup iterations
        print("【warm steps】", self.num_warm_steps)

        self.best_loss = np.inf
        self.best_val_loss = np.inf
        self.last_opt_step = -1
        if self.opt.skip_step1:
            self.get_epoch_iters()
            self.step = self.iters_every_epoch * self.opt.num_epochs
        if self.opt.start_epoch < self.opt.num_epochs:
            for self.epoch in range(self.opt.start_epoch, self.opt.num_epochs):
                if self.rank != -1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                if self.opt.skip_step1:
                    print("Skipping step1 training epochs (step1 already completed)", self.epoch)
                    continue
                self.run_epoch()

                if (self.epoch + 1) % self.opt.save_frequency == 0 and self.rank in [-1, 0] and not self.opt.save_best:
                    self.save_model()

        if self.opt.use_step_2:
            print("Step2 Training")
            self.set_step2()
            st_epoch = max(self.opt.num_epochs,self.opt.start_epoch )


            for self.epoch in range(st_epoch, self.opt.num_epochs + self.opt.step_2_epoch):
                self.run_epoch_step2()
                if (self.epoch + 1) % self.opt.save_frequency == 0 and self.rank in [-1, 0] and not self.opt.save_best:
                    self.save_model()

    def get_epoch_iters(self, ):
        self.iters_every_epoch = 0


        self.iters_every_epoch = math.ceil(len(self.train_dataset) / self.opt.batch_size)
        print("get epoch have num iter",self.iters_every_epoch)

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")

        self.set_train()
        self.max_grad = 0.0

        # Already checked, safe to disable AMP checks
        check_amp_pass = False
        iter_id = 0
        num_iters = np.inf
        self.use_detect_anomaly = False
        # Enable autograd anomaly detection during forward pass
        if self.use_detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        self.model_optimizer.zero_grad()


        # with torch.autograd.detect_anomaly():

        print("【use_iter】",self.use_iter)
        if self.use_dataprefetcher and not self.opt.all_mem:
            self.opt.device = self.device
            if hasattr(self, "prefetcher"):
                pass
                print("Already initialized, reloading new epoch data")
                self.prefetcher.re_load()
            else:
                prefetcher = DataPrefetcher(self.data, self.opt)
                self.prefetcher = prefetcher
            inputs = self.prefetcher.next()
        elif self.opt.all_mem:
            inputs = self.data[iter_id]
            num_iters = len(self.data)
        elif self.use_iter:
            self.data_iter = iter(self.data)
            inputs = next(self.data_iter)
        if self.use_iter:
            self.iter_epoch(self, inputs, num_iters,iter_id)
        else:
            self.default_epoch()


        self.model_lr_scheduler.step()
        if self.use_dataprefetcher and not self.opt.all_mem:
            # del prefetcher
            # print("Delete prefetcher to free VRAM")
            print("End Epoch")
            torch.cuda.empty_cache()
    def iter_epoch(self,inputs,num_iters,iter_id):
        while inputs is not None:
            iter_id += 1
            if iter_id >= num_iters:
                break
            if self.opt.for_debug:

                if iter_id > 100:
                    print("Early stop for debugging,", iter_id)
                    break

            batch_idx = iter_id

            before_op_time = time.time()
            ni = self.step
            if self.step == 1:
                # Only show model params and FLOPs on the first step
                input_color = inputs[("color", 0, 0)].cuda()
                print("[profile_once] Computing FLOPs and parameter count")

                #profile_once(self.models["encoder"], self.models["depth"], input_color)
                flops, params, flops_e, params_e, flops_d, params_d = profile_once(self.models["encoder"],
                                                                                   self.models["depth"], input_color,is_train=True)
                print(
                    "\n  " + ("flops: {0}, params: {1}, flops_e: {2}, params_e:{3}, flops_d:{4}, params_d:{5}").format(
                        flops,
                        params,
                        flops_e,
                        params_e,
                        flops_d,
                        params_d))
                self.model_info = {
                    'flops': flops,
                    'params': params,
                    'flops_e': flops_e,
                    'params_e': params_e,
                    'flops_d': flops_d,
                    'params_d': params_d
                }
                print(self.model_info)
            if self.step <= self.num_warm_steps:

                xi = [0, self.num_warm_steps]  # x interp
                self.accumulate = max(1, int(np.interp(ni, xi, [1, self.opt.nbs / self.opt.batch_size]).round()))
                for j, x in enumerate(self.model_optimizer.param_groups):
                    # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(
                        ni, xi, [self.opt.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(self.epoch)]
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [self.opt.warmup_momentum, self.opt.momentum])
            outputs, losses = self.process_batch(inputs)
            losses["loss"].backward()
            self.optimizer_step()

            if losses["loss"].cpu().data < self.best_loss and self.opt.save_best and self.rank in [-1,
                                                                                                   0] and self.step > 500:
                self.best_loss = losses["loss"].cpu().data
                print(f"Current best loss: {self.best_loss}, epoch:{self.epoch}, step:{self.step}")
                self.save_best_model(add_str="train")

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)

                max_grad = []
                for x in self.parameters_to_train:
                    max_grad.append(x.max())
                self.max_grad = max(max_grad).cpu().data.numpy()
                print("mem", mem, "max grad", self.max_grad)
                if losses != losses:
                    raise ValueError("NaN loss detected")



                self.log_time(batch_idx, duration, losses["loss"].cpu().data, self.max_grad)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                if self.rank in [-1, 0]:
                    self.log("train", inputs, outputs, losses)

                del inputs, outputs, losses
                if self.rank in [-1, 0]:
                    self.val()

            self.step += 1

            if self.use_dataprefetcher and not self.opt.all_mem:
                # prefetcher = DataPrefetcher(self.data, self.opt)
                inputs = self.prefetcher.next()
            elif self.opt.all_mem:
                inputs = self.data[iter_id]
                num_iters = len(self.data)
            else:
                try:
                    inputs = next(self.data_iter)
                except StopIteration:
                    inputs = None
                    break
    def default_epoch(self, ):
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            if self.step == 1:
                # Only show model params and FLOPs on the first step
                input_color = inputs[("color", 0, 0)].cuda()
                print("[profile_once] Computing FLOPs and parameter count")
                flops, params, flops_e, params_e, flops_d, params_d =  profile_once(self.models["encoder"], self.models["depth"], input_color,is_train=True)



                print(
                    "\n  " + ("flops: {0}, params: {1}, flops_e: {2}, params_e:{3}, flops_d:{4}, params_d:{5}").format(
                        flops,
                        params,
                        flops_e,
                        params_e,
                        flops_d,
                        params_d))
                self.model_info = {
                    'flops': flops,
                    'params': params,
                    'flops_e': flops_e,
                    'params_e': params_e,
                    'flops_d': flops_d,
                    'params_d': params_d
                }
                print(self.model_info)
            outputs, losses = self.process_batch(inputs)
            losses["loss"].backward()
            self.optimizer_step()



            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)

                max_grad = []
                for x in self.parameters_to_train:
                    max_grad.append(x.max())
                self.max_grad = max(max_grad).cpu().data.numpy()
                print("mem", mem, "max grad", self.max_grad)
                if losses != losses:
                    raise ValueError("NaN loss detected")

                self.log_time(batch_idx, duration, losses["loss"].cpu().data, self.max_grad)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                if self.rank in [-1, 0]:
                    self.log("train", inputs, outputs, losses)

                del inputs, outputs, losses
                if self.rank in [-1, 0]:
                    self.val()

            self.step += 1

    def run_epoch_step2(self):
        """Run a single epoch of training and validation
               """
        if self.opt.dataset == "cityscapes_preprocessed":
            if self.opt.num_epochs + 0 < self.epoch < self.opt.start_opt_epoch + self.opt.num_epochs and self.epoch % 5 ==1:
                model_path = os.path.join(str(self.log_path), "models", "weights_{}".format(self.epoch - 1))
                model_dict = self.models_f["encoder_0"].state_dict()
                encoder_dict = torch.load(os.path.join(model_path, "encoder.pth"))
                self.models_f["encoder_0"].load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
                print("Loading previous weights to depth_0, encoder_0", model_path)

                self.models_f["depth_0"].load_state_dict(
                    {k: v for k, v in torch.load(os.path.join(model_path, "depth.pth")).items() if
                     k in self.models_f["depth_0"].state_dict()}
                )

                self.models_f["depth_0"].to(self.device)
                self.models_f["encoder_0"].to(self.device)


        else:
            if self.opt.num_epochs + 0 < self.epoch < self.opt.start_opt_epoch + self.opt.num_epochs:

                model_path = os.path.join(str(self.log_path), "models", "weights_{}".format(self.epoch - 1))
                # model_path = os.path.join(str(self.log_path), "models", "weight_0")
                model_dict = self.models_f["encoder_0"].state_dict()
                encoder_dict = torch.load(os.path.join(model_path, "encoder.pth"))
                self.models_f["encoder_0"].load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
                print("Loading previous weights to depth_0, encoder_0", model_path)

                self.models_f["depth_0"].load_state_dict(
                    {k: v for k, v in torch.load(os.path.join(model_path, "depth.pth")).items() if
                     k in self.models_f["depth_0"].state_dict()}
                    )

                self.models_f["depth_0"].to(self.device)
                self.models_f["encoder_0"].to(self.device)

            """Run a single epoch of training and validation
            """

        print("【Training】 step2")
        self.set_train()
        self.max_grad = 0.0

        # Already checked, safe to disable AMP checks
        check_amp_pass = False
        iter_id = 0
        num_iters = np.inf
        self.use_detect_anomaly = False
        # Enable autograd anomaly detection during forward pass
        if self.use_detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        self.model_optimizer.zero_grad()

        # with torch.autograd.detect_anomaly():

        if self.use_dataprefetcher and not self.opt.all_mem:
            self.opt.device = self.device
            if hasattr(self, "prefetcher"):
                pass
                print("Already initialized, reloading new epoch data")
                self.prefetcher.re_load()
            else:
                prefetcher = DataPrefetcher(self.data, self.opt)
                self.prefetcher = prefetcher
            inputs = self.prefetcher.next()
        elif self.opt.all_mem:
            inputs = self.data[iter_id]
            num_iters = len(self.data)
        elif self.use_iter:
            self.data_iter = iter(self.data)
            inputs = next(self.data_iter)
            self.iter_epoch_step2(inputs,num_iters,iter_id)
        else:
            self.default_epoch_step2()


        self.model_lr_scheduler.step()
        if self.use_dataprefetcher and not self.opt.all_mem:
            # del prefetcher
            # print("Delete prefetcher to free VRAM")
            print("End Epoch")
            torch.cuda.empty_cache()

    def iter_epoch_step2(self,inputs,num_iters,iter_id):
        # for batch_idx, inputs in enumerate(self.data):
        while inputs is not None:
            iter_id += 1
            if iter_id >= num_iters:
                break
            if self.opt.for_debug:

                if iter_id > 100:
                    print("Early stop for debugging")
                    break

            batch_idx = iter_id

            before_op_time = time.time()
            ni = self.step
            if self.step <= self.num_warm_steps:

                xi = [0, self.num_warm_steps]  # x interp
                self.accumulate = max(1, int(np.interp(ni, xi, [1, self.opt.nbs / self.opt.batch_size]).round()))
                for j, x in enumerate(self.model_optimizer.param_groups):
                    # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(
                        ni, xi, [self.opt.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(self.epoch)]
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [self.opt.warmup_momentum, self.opt.momentum])

            outputs, losses = self.process_batch(inputs, step2=True)

            # with torch.autograd.detect_anomaly():
            losses["loss"].backward()
            self.optimizer_step()

            if losses["loss"].cpu().data < self.best_loss and self.opt.save_best and self.rank in [-1,
                                                                                                   0] and self.step > 500:
                self.best_loss = losses["loss"].cpu().data
                print(f"Current best loss: {self.best_loss}, epoch:{self.epoch}, step:{self.step}")
                self.save_best_model(add_str="train")
            duration = time.time() - before_op_time
            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)

                max_grad = []
                for x in self.parameters_to_train:
                    max_grad.append(x.max())
                # # print("max_grad",max(max_grad))
                self.max_grad = max(max_grad).cpu().data.numpy()
                print("mem", mem, "max grad", self.max_grad)
                if losses != losses:
                    raise ValueError("NaN loss detected")

                # self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.model_optimizer.param_groups)}  # for loggers

                self.log_time(batch_idx, duration, losses["loss"].cpu().data, self.max_grad)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                if self.rank in [-1, 0]:
                    self.log("train", inputs, outputs, losses, step2=True)

                del inputs, outputs, losses
                if self.rank in [-1, 0]:
                    self.val(step2=True)

            self.step += 1

            if self.use_dataprefetcher and not self.opt.all_mem:
                # prefetcher = DataPrefetcher(self.data, self.opt)
                inputs = self.prefetcher.next()
            elif self.opt.all_mem:
                inputs = self.data[iter_id]
                num_iters = len(self.data)
            else:
                try:
                    inputs = next(self.data_iter)
                except StopIteration:
                    inputs = None
                    break

    def default_epoch_step2(self):
        for batch_idx, inputs in enumerate(self.data):
            before_op_time = time.time()
            # self.model_optimizer.step()
            outputs, losses = self.process_batch(inputs, step2=True)
            # with torch.autograd.detect_anomaly():
            losses["loss"].backward()
            self.optimizer_step()
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)

                max_grad = []
                for x in self.parameters_to_train:
                    max_grad.append(x.max())
                self.max_grad = max(max_grad).cpu().data.numpy()
                print("mem", mem, "max grad", self.max_grad)
                if losses != losses:
                    raise ValueError("NaN loss detected")

                self.log_time(batch_idx, duration, losses["loss"].cpu().data, self.max_grad)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                if self.rank in [-1, 0]:
                    self.log("train", inputs, outputs, losses, step2=True)

                del inputs, outputs, losses
                if self.rank in [-1, 0]:
                    self.val(step2=True)
            self.step += 1
    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        if self.opt.amp:
            self.scaler.unscale_(self.model_optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(self.parameters_to_train, max_norm=10.0)  # clip gradients
            self.scaler.step(self.model_optimizer)
            self.scaler.update()
            self.model_optimizer.zero_grad()
        else:

            torch.nn.utils.clip_grad_norm_(self.parameters_to_train, max_norm=10.0)  # clip gradients
            self.model_optimizer.step()
            self.model_optimizer.zero_grad()

    def process_batch(self, inputs, step2=False):
        """Pass a minibatch through the network and generate images and losses
        """
        # print(inputs,"inputs")


        for key, ipt in inputs.items():  #

            if isinstance(ipt, int):
                ipt = torch.tensor(ipt)
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

            if step2:
                features_1 = self.models_f["encoder_0"](inputs["color_aug", 0, 0])
                outputs.update(self.models_f["depth_0"](features_1))
                features_2 = self.models_f["encoder"](inputs["color_aug", 0, 0])

                outputs.update(self.models_f["depth"](features_2))

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.opt.amp:
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                if self.use_pose_net:
                    outputs.update(self.predict_poses(inputs, features, step2=step2))
                # for amp
                if self.opt.amp:
                    for k, v in outputs.items():
                        if str(outputs[k].dtype) == "torch.float16":
                            outputs[k] = outputs[k].float()
        else:
            if self.use_pose_net:
                outputs.update(self.predict_poses(inputs, features, step2=step2))
        if self.opt.use_var_net and step2:
            var_inputs = torch.cat(
                [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)
            # print(self.models.keys())
            var_inputs = [self.models["var_encoder"](var_inputs)]
            outputs.update(self.models["var"](var_inputs[0]))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs, step2)

        return outputs, losses

    def _predict_poses_impl(self, inputs, features, step2=False):


        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":

                        if step2:
                            pose_inputs = [self.models_f["pose_encoder"](torch.cat(pose_inputs, 1))]
                        else:
                            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type in ["double_resnet", "double_resnet_all"]:
                        if step2:
                            pose_inputs = [self.models_f["pose_encoder"](torch.cat(pose_inputs, 1))]
                        else:
                            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "double_resnet_same":
                        if step2:

                            pose_inputs = [self.models_f["pose_encoder"](torch.cat(pose_inputs, 1))]
                        else:
                            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)
                    elif "yolo11" in self.opt.pose_model_type:

                        if step2:
                            pose_inputs = [self.models_f["pose_encoder"](torch.cat(pose_inputs, 1))]
                        else:
                            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                    if step2:
                        axisangle, translation = self.models_f["pose"](pose_inputs)
                    else:
                        axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle.float()
                    outputs[("translation", 0, f_i)] = translation.float()

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn",  "pose_encoder_same",
                                            ]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs


    def predict_poses(self, inputs, features, step2=False):
        """Predict poses between input frames for monocular sequences.
        """
        if self.opt.amp:
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                return self._predict_poses_impl(inputs, features, step2)
        else:
            return self._predict_poses_impl(inputs, features, step2)



    def val(self, step2=False):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)  # .next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)  # .next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs, step2=step2)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            if losses["loss"].cpu().data < self.best_val_loss and self.opt.save_best and self.rank in [-1,
                                                                                                       0] and self.step > 500:
                self.best_val_loss = losses["loss"].cpu().data
                print(f"Current best val loss: {self.best_val_loss}, epoch:{self.epoch}, step:{self.step}")
                self.save_best_model(add_str="val")
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """

        for scale in self.opt.scales:
        # For AMP compatibility to avoid NaN/Inf
            # print(outputs.keys())
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False).float()
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords.float()

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True).float()  # Compatibility for PyTorch >= 1.3.0

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)].float()

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            if self.opt.use_gmsd:
                pass
                gmsd_loss = self.gmsd(pred, target).mean(1, True)
                #fsim_loss = self.fsim(pred, target).mean(1, True)

                reprojection_loss = 0.7 * ssim_loss + 0.1 * l1_loss + 0.2 * gmsd_loss
            else:
                reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs, step2=False):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)].float()
            color = inputs[("color", 0, scale)].float()
            target = inputs[("color", 0, source_scale)].float()

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)].float()
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)].float()
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False).float()

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)
                if step2:
                    to_optimise = to_optimise.unsqueeze(1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()
            if self.opt.use_var_net and step2:
                var = outputs["var", scale]
                var_scale_0 = F.interpolate(
                    var, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                to_optimise = torch.log(var_scale_0 + 1) + to_optimise ** 2 / (2 * var_scale_0 ** 2)
            if step2:
                disp_f = outputs[("disp_f", scale)]
                disp_f_0 = outputs[("disp_f_0", scale)]


                beta = 0.7
                if self.opt.dy_mu:
                    # Using dynamic mu
                    # batch size
                    mu = self.models["MuPredictor"](outputs["feature_disp"],outputs["feature_disp_0"])
                    outputs["mu"] = mu
                    if random.random() < 0.001:
                        print("Random display: dynamic mu predicted value:", mu)

                else:
                    if self.opt.dataset == "cityscapes_preprocessed":
                        mu = 0.12
                    else:
                        mu = 0.1
                diff = torch.abs(disp_f_0.pow(beta) - disp_f.pow(beta))


                mask = diff < mu
                mask = mask.float()
                mask_0 = F.interpolate(
                    mask, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

                outputs["dynamic_mask/{}".format(scale)] = mask_0.squeeze()

                to_optimise *= mask_0

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            if step2:
                smooth_loss = get_smooth_loss(norm_disp, color, mask)
            else:
                smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80).float()
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"].float()
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask].float()
        depth_pred = depth_pred[mask].float()
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss, max_grad=0):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time



        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        if self.opt.world_size > 1:
            training_time_left = training_time_left / self.opt.world_size
        if max_grad > 0:
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                           " | loss: {:.5f} | time elapsed: {} | time left: {} | lr:{}"
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                      sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left), self.model_optimizer.param_groups[0]["lr"]))
        else:
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                           " | loss: {:.5f} | max grad: {:.3f}| time elapsed: {} | time left: {} | lr: {}"
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss, max_grad,
                                      sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left), self.model_optimizer.param_groups[0]["lr"]))

    def log(self, mode, inputs, outputs, losses, step2=False):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        metrics = {}
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
            metrics[l] = v

        if self.opt.use_wb:
            # Set wandb logging interval
            log_interval = getattr(self.opt, 'wb_log_interval', 50)  # Log every 50 steps by default

            if self.step % log_interval == 0:
                # Log scalar losses
                wandb_log_data = {f"{mode}/{l}": v for l, v in losses.items()}

                # If current step needs image logging
                # if self.step % (log_interval * 5) == 0:  # Lower image logging frequency, every 250 steps
                #     image_log_data = self._prepare_images_for_wandb(mode, inputs, outputs, step2)
                #     wandb_log_data.update(image_log_data)

                self.wb_run.log(wandb_log_data, step=self.step)
        if self.opt.use_wb and mode == "train" and hasattr(self, 'model_info') and not self.has_logged_model_info:
            # Log model info
            self.wb_run.log(self.model_info, step=self.step)
            self.has_logged_model_info = True  # Only log once



        if self.use_ray_tune:

            if ray.train.get_context().get_world_rank() == 0:
                ray_train.report(metrics, checkpoint=None)
        #         if self.opt.dy_mu and  step2:
        #
        #             for l, v in self.mu.items():
        #                 writer.add_scalar("{}".format(l), v, self.step)
        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images

            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)
                if step2:
                    if "mu" in outputs:
                        # Extract (B, 1, 1, 1) -> float
                        current_mu = outputs["mu"][j].item()
                        # Format mu value to string with 4 decimal places
                        mu_str = f"{current_mu:.4f}"

                        # Update dynamic mask title to include mu value
                        # New title format: "dynamic_mask_mu=0.1234_{}/{}".format(s, j)
                        writer.add_image(
                            f"dynamic_mask_mu={mu_str}_{s}/{j}",  # Use f-string for title
                            outputs["dynamic_mask/{}".format(s)][j][None, ...],
                            self.step)
                    else:
                        # If mu not found, use original title
                        writer.add_image(
                            "dynamic_mask{}/{}".format(s, j),
                            outputs["dynamic_mask/{}".format(s)][j][None, ...],
                            self.step)

                    if self.opt.use_var_net:
                        writer.add_image(
                            "var_d_{}/{}".format(s, j),
                            outputs[("var", s)][j].data, self.step)

    def _prepare_images_for_wandb(self, mode, inputs, outputs, step2):
        """Prepare image data for wandb logging, reduce storage"""
        image_data = {}
        batch_size_to_log = min(2, self.opt.batch_size)  # Limit logged images to 2

        for j in range(batch_size_to_log):
            for s in self.opt.scales:
                # Only log scale=0 images to reduce data
                if s > 0:
                    continue

                for frame_id in self.opt.frame_ids:
                    # Convert tensor to wandb image format
                    color_img = inputs[("color", frame_id, s)][j].data
                    if color_img.dim() == 3:
                        color_img_np = color_img.cpu().numpy().transpose(1, 2, 0)
                        # Normalize to 0-255
                        if color_img_np.max() <= 1.0:
                            color_img_np = (color_img_np * 255).astype('uint8')
                        image_data[f"{mode}/color_{frame_id}_{s}/{j}"] = wandb.Image(color_img_np)

                    if s == 0 and frame_id != 0:
                        color_pred_img = outputs[("color", frame_id, s)][j].data
                        if color_pred_img.dim() == 3:
                            color_pred_img_np = color_pred_img.cpu().numpy().transpose(1, 2, 0)
                            if color_pred_img_np.max() <= 1.0:
                                color_pred_img_np = (color_pred_img_np * 255).astype('uint8')
                            image_data[f"{mode}/color_pred_{frame_id}_{s}/{j}"] = wandb.Image(color_pred_img_np)

                # Disparity maps
                disp_img = normalize_image(outputs[("disp", s)][j])
                if disp_img.dim() == 3:
                    disp_img_np = disp_img.detach().cpu().numpy().transpose(1, 2, 0)
                    if disp_img_np.max() <= 1.0:
                        disp_img_np = (disp_img_np * 255).astype('uint8')
                image_data[f"{mode}/disp_{s}/{j}"] = wandb.Image(disp_img)

                # Other image types...
                if step2 and "mu" in outputs:
                    current_mu = outputs["mu"][j].item()
                    mu_str = f"{current_mu:.4f}"

                    if "dynamic_mask/{}".format(s) in outputs:
                        mask_img = outputs["dynamic_mask/{}".format(s)][j][None, ...]
                        if mask_img.dim() == 3:
                            mask_img_np = mask_img.detach().cpu().numpy().transpose(1, 2, 0)
                            if mask_img_np.max() <= 1.0:
                                mask_img_np = (mask_img_np * 255).astype('uint8')
                            image_data[f"{mode}/dynamic_mask_mu={mu_str}_{s}/{j}"] = wandb.Image(mask_img_np)

        return image_data
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)
        if self.opt.use_wb:
            custom_save_dir = os.path.join(self.log_path,"wandb_log")
            if not os.path.exists(custom_save_dir):
                os.makedirs(custom_save_dir)
            

            # Set WANDB_API_KEY via environment variable before running, e.g.:
            # export WANDB_API_KEY=your_key_here
            if 'WANDB_API_KEY' not in os.environ:
                print("Warning: WANDB_API_KEY not set. Set it via: export WANDB_API_KEY=your_key")

            print("wandb save",custom_save_dir)
            # wb_config['mode'] = 'offline'

            os.environ['WANDB_DIR'] = custom_save_dir

            project="FlexDepth"
            wandb.login()
            self.wb_run = wandb.init(project=project, config=to_save,)# mode='offline'



        self.save_py()

    def save_py(self):
        '''
        """Back up current code for each experiment run"""
        :return:
        :rtype:
        '''
        import shutil
        models_dir_ = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir_):
            os.makedirs(models_dir_)

        models_dir = os.path.join(models_dir_,"codeSnapshot")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # If target directory doesn't exist, create it
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

            # Traverse current directory and subdirectories
        copy_files = []
        not_copys = []
        for root, dirs, files in os.walk(current_dir):
            # Exclude logs folder
            if "logs" in root.split(os.sep):
                continue

            # Process each .py file
            for file in files:
                if file.endswith('.py'):
                    # Get source file full path
                    src_file = os.path.join(root, file)
                    # Compute relative path
                    rel_path = os.path.relpath(src_file, current_dir)
                    # Compute target file full path
                    dst_file = os.path.join(models_dir, rel_path)
                    # Ensure target directory exists
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    # Copy file
                    try:
                        shutil.copy(src_file, dst_file)
                        copy_files.append(src_file)
                    except Exception as e:
                        not_copys.append(src_file)


        print(f"All Python files copied to {models_dir}")
        print("Total files copied:", len(copy_files), copy_files[:3], "...")
        print("not_copys",not_copys)



    def save_model(self):
        """Save model weights to disk
        """
        st = time.time()

        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = de_parallel(model).state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
                to_save['decoder_model_type'] = self.opt.decoder_model_type
                to_save['c3k'] = self.opt.c3k
                to_save['all_dysample'] = self.opt.all_dysample
                to_save['dysample_group8'] = self.opt.dysample_group8
                to_save['upsample'] = self.opt.upsample
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
        ed = time.time()
        print("Model save time:", ed - st)

    def save_best_model(self, add_str="train"):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_best{}".format(add_str))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = de_parallel(model).state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
                to_save['decoder_model_type'] = self.opt.decoder_model_type
                to_save['c3k'] = self.opt.c3k
                to_save['all_dysample'] = self.opt.all_dysample
                to_save['dysample_group8'] = self.opt.dysample_group8
                to_save['upsample'] = self.opt.upsample
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self,step2=False):
        """Load model(s) from disk
        """
        with torch_distributed_zero_first(self.rank):
            self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

            assert os.path.isdir(self.opt.load_weights_folder), \
                "Cannot find folder {}".format(self.opt.load_weights_folder)
            print("loading model from folder {}".format(self.opt.load_weights_folder))

            for n in self.opt.models_to_load:
                if step2 and n in ["pose_encoder","pose"]:
                    print("STEP2 CONTINUE ",n)
                    continue
                print("Loading {} weights...".format(n))
                path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))

                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

            # loading adam state
            optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
            if os.path.isfile(optimizer_load_path):
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            else:
                print("Cannot find Adam weights so Adam is randomly initialized")
