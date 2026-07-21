import datetime
import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

try:
    from thop import clever_format
    from thop import profile
except ImportError:
    thop = None
logger = logging.getLogger(__name__)

print("【import thop 】",thop )


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False
def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model

def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # not a git repository


def select_device(device='', batch_size=None,local_rank=-1):
    # device = 'cpu' or '0' or '0,1,2,3'
    if local_rank > 0:
        # torch.cuda.set_device(local_rank)
        print(torch.cuda.device(local_rank), "Selected device")
        cpu = device.lower() == 'cpu'
        cuda = not cpu and torch.cuda.is_available()
        return torch.device(f'cuda:{local_rank}' if cuda else 'cpu')

    if isinstance(device, torch.device):
        return device
    device = str(device).lower()
    s = f'monodepth 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'
    print(s,)
    print("")

    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(x, ops, n=100, device=None):
    # profile a pytorch module or list of modules. Example usage:
    #     x = torch.randn(16, 3, 640, 640)  # input
    #     m1 = lambda x: x * torch.sigmoid(x)
    #     m2 = nn.SiLU()
    #     profile(x, [m1, m2], n=100)  # profile speed over 100 iterations

    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    x.requires_grad = True
    print(torch.__version__, device.type, torch.cuda.get_device_properties(0) if device.type == 'cuda' else '')
    print(f"\n{'Params':>12s}{'GFLOPS':>12s}{'forward (ms)':>16s}{'backward (ms)':>16s}{'input':>24s}{'output':>24s}")
    for m in ops if isinstance(ops, list) else [ops]:
        m = m.to(device) if hasattr(m, 'to') else m  # device
        m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m  # type
        dtf, dtb, t = 0., 0., [0., 0., 0.]  # dt forward, backward
        try:
            flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPS
        except:
            flops = 0

        for _ in range(n):
            t[0] = time_synchronized()
            y = m(x)
            t[1] = time_synchronized()
            try:
                _ = y.sum().backward()
                t[2] = time_synchronized()
            except:  # no backward method
                t[2] = float('nan')
            dtf += (t[1] - t[0]) * 1000 / n  # ms per op forward
            dtb += (t[2] - t[1]) * 1000 / n  # ms per op backward

        s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else 'list'
        s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else 'list'
        p = sum(list(x.numel() for x in m.parameters())) if isinstance(m, nn.Module) else 0  # parameters
        print(f'{p:12}{flops:12.4g}{dtf:16.4g}{dtb:16.4g}{str(s_in):>24s}{str(s_out):>24s}')


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
            #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # Return global model sparsity
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv




def model_info(model, detailed=False,verbose=False, img_size=640,num_img=1):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    print("img_size",img_size)
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose or detailed:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))



    try:  # FLOPS
        from thop import profile  as thop_profile
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        img = torch.zeros((1, 3*num_img, img_size[0], img_size[1]), device=next(model.parameters()).device)  # input
        flops = thop_profile(deepcopy(model), inputs=img, verbose=False)[0] / 1E9  # stride GFLOPS
        fs = '%.1f GFLOPS' % (flops)  # 640x640 GFLOPS

        print(fs)
    except (ImportError, Exception):
        print("THOP NOT EXISTS")
        flops = get_flops_with_torch_profiler(model,imgsz=img_size)
        fs = '%.1f GFLOPS' % (flops)


    logger.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients, {fs}")
    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients, {fs}")



def moudle_info(moudle,input_x, detailed=False,verbose=False,):
    # Moudle information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in moudle.parameters())  # number parameters
    n_g = sum(x.numel() for x in moudle.parameters() if x.requires_grad)  # number gradients
    if verbose or detailed:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(moudle.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))



    try:  # FLOPS
        from thop import profile  as thop_profile
        print("THOP  EXISTS")

        flops = thop_profile(deepcopy(moudle), inputs=input_x, verbose=False)[0] / 1E9  # stride GFLOPS
        fs = '%.1f GFLOPS' % (flops)  #  GFLOPS
        print("Comparing two computation methods thop_profile", fs)
        try:
            flops = get_flops_with_torch_profiler_moudle(moudle, input_x=input_x)
            fs = '%.1f GFLOPS' % (flops)
            print("Comparing two computation methods torch.profiler.profile", fs)
        except Exception as e:
            print("line 289 e:",e)
            import traceback
            traceback.print_exc()
            print("Comparing two computation methods thop_profile", fs)


    except (ImportError, Exception):
        print("THOP NOT EXISTS")

        flops = get_flops_with_torch_profiler_moudle(moudle,input_x=input_x)
        fs = '%.1f GFLOPS' % (flops)


    logger.info(f"Moudle Summary: {len(list(moudle.modules()))} layers, {n_p} parameters, {n_g} gradients, {fs}")
    print(f"Moudle Summary: {len(list(moudle.modules()))} layers, {n_p} parameters, {n_g} gradients, {fs}")

    return flops, n_p


def profile_once(encoder, decoder, x,is_train=False):
    x_e = x[0, :, :, :].unsqueeze(0)

    x_d = encoder(x_e)
    try:
        if is_train:
            # Gradient info during training may cause errors, so don't deep copy
            flops_e, params_e = thop.profile(encoder, inputs=(x_e, ), verbose=False)
            flops_d, params_d = thop.profile(decoder, inputs=(x_d, ), verbose=False)
        else:
            flops_e, params_e = thop.profile(deepcopy(encoder), inputs=(x_e,), verbose=False)
            flops_d, params_d = thop.profile(deepcopy(decoder), inputs=(x_d,), verbose=False)

        flops, params = clever_format([flops_e + flops_d, params_e + params_d], "%.3f")
        flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
        flops_d, params_d = clever_format([flops_d, params_d], "%.3f")
        if is_train:
            pass
        else:
            if is_train:
                print("【Encoder】")

                flops_e_, params_e_ = moudle_info(encoder, (x_e,))
                print("【Decoder】")
                flops_d_, params_d_ = moudle_info(decoder, (x_d,))
                flops_, params_ = flops_e_ + flops_d_, params_e_ + params_d_
            else:
                print("【Encoder】")

                flops_e_, params_e_ = moudle_info(deepcopy(encoder), (x_e,))
                print("【Decoder】")
                flops_d_, params_d_ = moudle_info(deepcopy(decoder), (x_d,))
                flops_, params_ = flops_e_ + flops_d_, params_e_ + params_d_


    except Exception as e:
        import traceback
        print("error",e)
        traceback.print_exc()
        if is_train:

            print("【Encoder】")
            flops_e, params_e = moudle_info(encoder,(x_e, ))
            print("【Decoder】")
            flops_d, params_d = moudle_info(decoder, (x_d, ))
            flops , params = flops_e + flops_d ,params_e + params_d
        else:
            print("【Encoder】")
            flops_e, params_e = moudle_info(deepcopy(encoder), (x_e,))
            print("【Decoder】")
            flops_d, params_d = moudle_info(deepcopy(decoder), (x_d,))
            flops, params = flops_e + flops_d, params_e + params_d

    return flops, params, flops_e, params_e, flops_d, params_d



def get_flops_with_torch_profiler(model, imgsz=640):
    """Compute model FLOPs (thop package alternative, but 2-10x slower unfortunately)."""

    model = de_parallel(model)
    p = next(model.parameters())
    if not isinstance(imgsz, list):
        imgsz = [imgsz, imgsz]  # expand if int/float
    print("imgsz",imgsz)
    # try:
    #     # Use stride size for input tensor
    #     stride = (max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32) * 2  # max stride
    #     im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
    #     with torch.profiler.profile(with_flops=True) as prof:
    #         model(im)
    #     flops = sum(x.flops for x in prof.key_averages()) / 1e9
    #     flops = flops * imgsz[0] / stride * imgsz[1] / stride  # 640x640 GFLOPs
    #     print("Use stride size for input tensor")
    # except Exception:
    #     # Use actual image size for input tensor (i.e. required for RTDETR models)
    #
    #     im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # input image in BCHW format
    #
    #     with torch.profiler.profile(with_flops=True) as prof:
    #         model(im)
    #     flops = sum(x.flops for x in prof.key_averages()) / 1e9
    im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # input image in BCHW format

    with torch.profiler.profile(with_flops=True) as prof:
        model(im)
    flops = sum(x.flops for x in prof.key_averages()) / 1e9
    print("image shape test profile", im.shape)
    return flops


def get_flops_with_torch_profiler_moudle(moudle, input_x):
    """Compute model FLOPs (thop package alternative, but 2-10x slower unfortunately)."""

    model = de_parallel(moudle)
    print("Cleaning up existing forward/pre-forward hooks from model...")
    for module in model.modules():
        # Clear forward hooks
        if hasattr(module, '_forward_hooks') and module._forward_hooks:
            module._forward_hooks.clear()
        # Clear pre-forward hooks
        if hasattr(module, '_forward_pre_hooks') and module._forward_pre_hooks:
            module._forward_pre_hooks.clear()
    p = next(moudle.parameters())
    if isinstance(input_x,tuple):
        input_x = input_x[0]

    if isinstance(input_x, list):
        im = []  # Initialize list for dummy input tensors
        for tensor in input_x:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Each element in 'input_x' list must be a torch.Tensor.")
            im.append(torch.empty_like(tensor, device=p.device))  # Create dummy tensor with same shape/dtype/device
        print("Profiling FLOPs with input shapes (list):", [tensor.shape for tensor in im])

    else:
        im = torch.empty_like(input_x, device=p.device)
    if isinstance(im,torch.Tensor):
        print("Profiling FLOPs with input shape:", im.shape)  #

    with torch.profiler.profile(with_flops=True) as prof:
        model(im)
    flops = sum(x.flops for x in prof.key_averages()) / 1e9
    if isinstance(im, torch.Tensor):
        print("image shape test profile", im.shape)
    return flops
def load_classifier(name='resnet101', n=2):
    # Loads a pretrained model reshaped to n-class output
    model = torchvision.models.__dict__[name](pretrained=True)

    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False).float()  # resize
        if not same_shape:  # pad/crop img
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)



def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Returns a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf."""
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1

class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)