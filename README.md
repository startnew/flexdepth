# FlexDepth

**Towards Robust Driving Perception: A Flexible Scale-Driven Family for Self-Supervised Monocular Depth Estimation**

[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-4F70F2?style=flat-square)](https://eccv.ecva.net/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.00736-b31b1b?style=flat-square)](https://arxiv.org/abs/2607.00736)
[![Project Page](https://img.shields.io/badge/Project-Page-4F70F2?style=flat-square)](https://startnew.github.io/projects/flexdepth/)
![Visitors](https://api.visitorbadge.io/api/visitors?path=startnew.flexdepth&label=Visitors&countColor=%23263759&style=flat-square)

---

@[TOC]

## News

- **[2026-07]** Code is now available!
- **[2026-07]** Project page is live at [startnew.github.io/projects/flexdepth](https://startnew.github.io/projects/flexdepth/)
- **[2026-06]** Accepted by ECCV 2026

## Links

- [Paper (arXiv)](https://arxiv.org/abs/2607.00736)
- [Project Page](https://startnew.github.io/projects/flexdepth/)
- [Video Results](https://startnew.github.io/projects/flexdepth/#comparison)
- [Google Drive](https://drive.google.com/drive/folders/1sOp04-zCwkC3JJN9gMbu2GbjUdAJfp6r?usp=sharing) / [HuggingFace](https://huggingface.co/StarNew/flexdepth) /[Baidu Netdisk](https://pan.baidu.com/s/1U5vtDhDr2WH3v6L6NeNKKA?pwd=zncb).

## Installation

```bash
conda create -n flexdepth python=3.10
conda activate flexdepth

# PyTorch (adjust CUDA version as needed)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# Dependencies
pip install -r requirements.txt
```

The `requirements.txt` includes `ultralytics`, `timm`, `prefetch_generator`, `wandb`, `tensorboardX`, and other dependencies used in this project.

## Prepare Datasets

### KITTI

**Important: We use PNG images directly. Skip the JPG conversion step when following Monodepth2's data preparation.**

Follow [Monodepth2](https://github.com/nianticlabs/monodepth2) to download the KITTI dataset. The default data path is `./kitti_data_png`, or specify via `--data_path`.

### Cityscapes

Follow [Manydepth](https://github.com/nianticlabs/manydepth) or [DynamicDepth](https://github.com/AutoAILab/DynamicDepth) to download and preprocess the Cityscapes dataset. Specify the path via `--data_path <cityscapes_path>`.

### Pretrained YOLO11 Weights

Download YOLO11 segmentation pretrained weights from [Ultralytics](https://github.com/ultralytics/assets/releases/tag/v8.3.0) and place them in `./ckpt/`:

| Scale | Download |
|-------|----------|
| Nano | [yolo11n-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt) |
| Small | [yolo11s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt) |
| Medium | [yolo11m-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt) |
| Large | [yolo11l-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt) |
| X-Large | [yolo11x-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt) |

> **Tip:** Since the YOLO encoder structure remains unchanged from YOLO11 to YOLO26, you can also use YOLO26 COCO segmentation pretrained weights for encoder initialization, which may yield better results. Adjust hyperparameters accordingly.



## Training

> **Note:** Training code (`train.py`, `trainer.py`) is not included in this release but will be available soon. The commands below are provided for reference when the training code is released.

### KITTI

> **Note:** Our reported results were obtained with N/S/M/L models trained on an RTX 2080 Ti and X-Large on an RTX 4090. Adjust `--batch_size` according to your GPU memory.

**Two-stage training in one command** (recommended):

```bash
# Flex-Nano (2080 Ti, lr=1e-4, bs=12)
python train.py --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 20 \
    --resume --scale 4 --optim NAdam --learning_rate 1e-4 \
    --encoder_model_type yolo11n-seg --decoder_model_type flexn --batch_size 12 \
    --height 192 --width 640 --dy_mu --png

# Flex-Small (2080 Ti, lr=1e-4, bs=12)
python train.py --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 20 \
    --resume --scale 4 --optim NAdam --learning_rate 1e-4 \
    --encoder_model_type yolo11s-seg --decoder_model_type flexs --batch_size 12 \
    --height 192 --width 640 --dy_mu --png

# Flex-Medium (2080 Ti, lr=5e-5, bs=6)
python train.py --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 20 \
    --resume --scale 4 --optim NAdam --learning_rate 5e-5 \
    --encoder_model_type yolo11m-seg --decoder_model_type flexm --batch_size 6 \
    --height 192 --width 640 --dy_mu --png

# Flex-Large (2080 Ti, lr=5e-5, bs=6)
python train.py --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 20 \
    --resume --scale 4 --optim NAdam --learning_rate 5e-5 \
    --encoder_model_type yolo11l-seg --decoder_model_type flexl --batch_size 6 \
    --height 192 --width 640 --dy_mu --png

# Flex-X-Large (4090, lr=5e-5, bs=12)
python train.py --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 20 \
    --resume --scale 4 --optim NAdam --learning_rate 5e-5 \
    --encoder_model_type yolo11x-seg --decoder_model_type flexx --batch_size 12 \
    --height 192 --width 640 --dy_mu --png
```

**Or train two stages separately:**

```bash
# Stage 1 only (example: Flex-X-Large)
python train.py --num_epochs 30 --resume --scale 4 --optim NAdam \
    --learning_rate 5e-5 --encoder_model_type yolo11x-seg --decoder_model_type flexx \
    --batch_size 12 --height 192 --width 640 --dy_mu --png

# Stage 2 only (skip stage 1)
python train.py --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 20 \
    --resume --scale 4 --optim NAdam --learning_rate 5e-5 \
    --encoder_model_type yolo11x-seg --decoder_model_type flexx --batch_size 12 \
    --height 192 --width 640 --dy_mu --png --skip_step1
```

### Cityscapes

```bash
# Flex-Nano
python train.py --dataset cityscapes_preprocessed --split cityscapes_preprocessed \
    --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 10 \
    --resume --scale 4 --optim NAdam --learning_rate 1e-4 \
    --encoder_model_type yolo11n-seg --decoder_model_type flexn --batch_size 12 \
    --height 192 --width 512 --dy_mu --data_path <cityscapes_path>

# Flex-Small
python train.py --dataset cityscapes_preprocessed --split cityscapes_preprocessed \
    --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 10 \
    --resume --scale 4 --optim NAdam --learning_rate 1e-4 \
    --encoder_model_type yolo11s-seg --decoder_model_type flexs --batch_size 12 \
    --height 192 --width 512 --dy_mu --data_path <cityscapes_path>

# Flex-Medium
python train.py --dataset cityscapes_preprocessed --split cityscapes_preprocessed \
    --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 10 \
    --resume --scale 4 --optim NAdam --learning_rate 1e-4 \
    --encoder_model_type yolo11m-seg --decoder_model_type flexm --batch_size 6 \
    --height 192 --width 512 --dy_mu --png --data_path <cityscapes_path>

# Flex-Large
python train.py --dataset cityscapes_preprocessed --split cityscapes_preprocessed \
    --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 10 \
    --resume --scale 4 --optim NAdam --learning_rate 1e-4 \
    --encoder_model_type yolo11l-seg --decoder_model_type flexl --batch_size 6 \
    --height 192 --width 512 --dy_mu --png --data_path <cityscapes_path>

# Flex-X-Large
python train.py --dataset cityscapes_preprocessed --split cityscapes_preprocessed \
    --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 10 \
    --resume --scale 4 --optim NAdam --learning_rate 5e-5 \
    --encoder_model_type yolo11x-seg --decoder_model_type flexx --batch_size 6 \
    --height 192 --width 512 --dy_mu --png --data_path <cityscapes_path>
```

## Evaluation

### KITTI

```bash
# Flex-Nano
python evaluate_depth.py --png --eval_mono --scale 4 \
    --encoder_model_type yolo11n-seg --decoder_model_type flexn \
    --load_weights_folder ./models/kitti/flex_n \
    --data_path <kitti_data_path> --split_path <splits_path>

# Flex-Small
python evaluate_depth.py --png --eval_mono --scale 4 \
    --encoder_model_type yolo11s-seg --decoder_model_type flexs \
    --load_weights_folder ./models/kitti/flex_s \
    --data_path <kitti_data_path> --split_path <splits_path>

# Flex-Medium
python evaluate_depth.py --png --eval_mono --scale 4 \
    --encoder_model_type yolo11m-seg --decoder_model_type flexm \
    --load_weights_folder ./models/kitti/flex_m \
    --data_path <kitti_data_path> --split_path <splits_path>

# Flex-Large
python evaluate_depth.py --png --eval_mono --scale 4 \
    --encoder_model_type yolo11l-seg --decoder_model_type flexl \
    --load_weights_folder ./models/kitti/flex_l \
    --data_path <kitti_data_path> --split_path <splits_path>

# Flex-X-Large
python evaluate_depth.py --png --eval_mono --scale 4 \
    --encoder_model_type yolo11x-seg --decoder_model_type flexx \
    --load_weights_folder ./models/kitti/flex_x \
    --data_path <kitti_data_path> --split_path <splits_path>
```

**Results on KITTI Eigen split (Cap 80m):**

| Model | Params | GFLOPs | Abs Rel ↓ | Sq Rel ↓ | RMSE ↓ | RMSE log ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ |
|-------|--------|--------|-----------|----------|--------|------------|----------|-----------|-----------|
| Flex-Nano | 1.5M | 0.7 | 0.110 | 0.794 | 4.678 | 0.184 | 0.878 | 0.961 | 0.983 |
| Flex-Small | 6.1M | 2.8 | 0.104 | 0.713 | 4.458 | 0.179 | 0.890 | 0.964 | 0.983 |
| Flex-Medium | 12.7M | 10.0 | 0.096 | 0.639 | 4.253 | 0.172 | 0.903 | 0.968 | 0.985 |
| Flex-Large | 15.2M | 11.5 | 0.095 | 0.642 | 4.199 | 0.171 | 0.906 | 0.968 | 0.984 |
| Flex-X-Large | 32.3M | 24.6 | **0.093** | **0.605** | **4.114** | **0.167** | **0.910** | **0.969** | **0.985** |

### Comparison with Depth Anything 2 (Dense GT, Least-Squares Alignment)

To compare with Depth Anything 2 and other zero-shot depth models, we evaluate Flex-X-Large on the KITTI Eigen benchmark split using **dense ground truth** with **least-squares alignment** (instead of the median scaling alignment used in the standard evaluation above). This aligns with the evaluation protocol used by DA2.

First, generate the dense ground truth (following [Monodepth2](https://github.com/nianticlabs/monodepth2)):

```bash
python export_gt_depth.py --data_path <kitti_data_path> --split eigen_benchmark
```

Then evaluate:

```bash
python evaluate_depth.py --png --eval_mono --scale 4 \
    --encoder_model_type yolo11x-seg --decoder_model_type flexx \
    --load_weights_folder ./models/kitti/flex_x \
    --data_path <kitti_data_path> --split_path <splits_path> \
    --eval_split eigen_benchmark --use_lstsq_alignment
```

**Results on KITTI Eigen benchmark (Dense GT, Least-Squares Alignment):**

| Method | Type | Params | GFLOPs | Resolution | Abs Rel ↓ | δ<1.25 ↑ |
|--------|------|--------|--------|------------|-----------|----------|
| DA2 (ViT-L) | Zero-Shot | 335M | 1947 | 1722×518 | 0.070 | **0.956** |
| DA2 (ViT-S) | Zero-Shot | 25M | 137 | 1722×518 | 0.077 | 0.944 |
| DA2 (ViT-L) | Zero-Shot | 335M | 276 | 644×196 | 0.092 | 0.915 |
| DA2 (ViT-S) | Zero-Shot | 25M | 19 | 644×196 | 0.110 | 0.881 |
| Flex-X-Large (Ours) | Self-Supervised | 32M | 25 | 640×192 | **0.063** | 0.952 |

### Cityscapes

```bash
# Flex-Nano
python evaluate_depth.py --eval_mono --scale 4 \
    --dataset cityscapes_preprocessed --eval_split cityscapes \
    --encoder_model_type yolo11n-seg --decoder_model_type flexn \
    --load_weights_folder ./models/cs/flex_n \
    --data_path <cityscapes_path> --split_path <splits_path>

# Flex-Small
python evaluate_depth.py --eval_mono --scale 4 \
    --dataset cityscapes_preprocessed --eval_split cityscapes \
    --encoder_model_type yolo11s-seg --decoder_model_type flexs \
    --load_weights_folder ./models/cs/flex_s \
    --data_path <cityscapes_path> --split_path <splits_path>

# Flex-Medium
python evaluate_depth.py --eval_mono --scale 4 \
    --dataset cityscapes_preprocessed --eval_split cityscapes \
    --encoder_model_type yolo11m-seg --decoder_model_type flexm \
    --load_weights_folder ./models/cs/flex_m \
    --data_path <cityscapes_path> --split_path <splits_path>

# Flex-Large
python evaluate_depth.py --eval_mono --scale 4 \
    --dataset cityscapes_preprocessed --eval_split cityscapes \
    --encoder_model_type yolo11l-seg --decoder_model_type flexl \
    --load_weights_folder ./models/cs/flex_l \
    --data_path <cityscapes_path> --split_path <splits_path>

# Flex-X-Large
python evaluate_depth.py --eval_mono --scale 4 \
    --dataset cityscapes_preprocessed --eval_split cityscapes \
    --encoder_model_type yolo11x-seg --decoder_model_type flexx \
    --load_weights_folder ./models/cs/flex_x \
    --data_path <cityscapes_path> --split_path <splits_path>
```

**Results on Cityscapes (During evaluation, crop follow manydepth,pro depth etc.):**

| Model | Params | GFLOPs | Abs Rel ↓ | Sq Rel ↓ | RMSE ↓ | RMSE log ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ |
|-------|--------|--------|-----------|----------|--------|------------|----------|-----------|-----------|
| Flex-Nano | 1.5M | 0.6 | 0.107 | 1.261 | 6.133 | 0.164 | 0.893 | 0.971 | 0.989 |
| Flex-Small | 6.1M | 2.2 | 0.100 | 1.078 | 5.813 | 0.153 | 0.904 | 0.975 | 0.991 |
| Flex-Medium | 12.7M | 8.0 | 0.089 | 0.885 | 5.358 | 0.143 | 0.917 | 0.979 | 0.993 |
| Flex-Large | 15.2M | 9.2 | 0.087 | 0.911 | 5.310 | 0.139 | 0.924 | 0.981 | 0.993 |
| Flex-X-Large | 32.3M | 19.7 | **0.086** | **0.877** | **5.268** | **0.137** | **0.926** | **0.982** | **0.993** |

## Pretrained Models

Pretrained model weights are **not included** in this repository due to file size. Download them separately and place them in `./models/`.

Expected directory structure after download:
```
models/
├── kitti/
│   ├── flex_n/
│   │   ├── depth.pth
│   │   └── encoder.pth
│   ├── flex_s/
│   ├── flex_m/
│   ├── flex_l/
│   └── flex_x/
└── cs/
    ├── flex_n/
    ├── flex_s/
    ├── flex_m/
    ├── flex_l/
    └── flex_x/
```

Weights  available  via [Google Drive](https://drive.google.com/drive/folders/1sOp04-zCwkC3JJN9gMbu2GbjUdAJfp6r?usp=sharing) / [HuggingFace](https://huggingface.co/StarNew/flexdepth).

### What is included in this repository

This repository contains only the source code, configuration files, and dataset split lists required to train, evaluate, and export the models:

- Source code: `options.py`, `evaluate_depth.py`, `export_onnx.py`, `layers.py`, `utils.py`, `kitti_utils.py`, etc.
- Network definitions: `networks/`, `layers.py`
- Dataset loaders and splits: `datasets/`, `splits/`
- YOLO encoder config: `cfg/models/yolo11-dep-encoder.yaml`
- Utilities: `utils_add/`, `utils.py`, `kitti_utils.py`
- Setup files: `requirements.txt`, `.gitignore`, `LICENSE`, `README.md`

The following are **excluded** and must be downloaded separately:

- KITTI / Cityscapes / Make3D datasets
- YOLO11 pretrained weights (place under `./ckpt/`)
- FlexDepth trained weights (place under `./models/`)
- Training logs and exported ONNX files

## ONNX Export

```bash
# Flex-Nano
python export_onnx.py --encoder_model_type yolo11n-seg --decoder_model_type flexn \
    --load_weights_folder ./models/kitti/flex_n --scales 4 --export_name flex-n

# Flex-Small
python export_onnx.py --encoder_model_type yolo11s-seg --decoder_model_type flexs \
    --load_weights_folder ./models/kitti/flex_s --scales 4 --export_name flex-s

# Flex-Medium
python export_onnx.py --encoder_model_type yolo11m-seg --decoder_model_type flexm \
    --load_weights_folder ./models/kitti/flex_m --scales 4 --export_name flex-m

# Flex-Large
python export_onnx.py --encoder_model_type yolo11l-seg --decoder_model_type flexl \
    --load_weights_folder ./models/kitti/flex_l --scales 4 --export_name flex-l

# Flex-X-Large
python export_onnx.py --encoder_model_type yolo11x-seg --decoder_model_type flexx \
    --load_weights_folder ./models/kitti/flex_x --scales 4 --export_name flex-x
```

## Citation

```bibtex
@misc{zhu2026robustdrivingperceptionflexible,
  title={Towards Robust Driving Perception: A Flexible Scale-Driven Family for Self-Supervised Monocular Depth Estimation},
  author={Zhaowen Zhu and Li Zhang and Yujie Chen and Tian Zhang and Yingjie Wang and Mingxia Zhan},
  year={2026},
  eprint={2607.00736},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2607.00736}
}
```

## Acknowledgment

This work is supported by the National Natural Science Foundation of China under Grant 62332016.

Our code is built upon [Monodepth2](https://github.com/nianticlabs/monodepth2), [Manydepth](https://github.com/nianticlabs/manydepth), and [Ultralytics](https://github.com/ultralytics/ultralytics).

We thank [DynamicDepth](https://github.com/AutoAILab/DynamicDepth) for providing dynamic scene annotations on Cityscapes, [DiPE](https://github.com/HalleyJiang/DiPE/tree/main) for providing dynamic scene annotations on KITTI, and [DSI-training](https://github.com/zhangtian33/DSI-training) for the masking approach.
