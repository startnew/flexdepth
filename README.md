# FlexDepth

**Towards Robust Driving Perception: A Flexible Scale-Driven Family for Self-Supervised Monocular Depth Estimation**

[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-4F70F2?style=flat-square)](https://link.springer.com/chapter/10.1007/978-3-032-36846-1_7)
[![arXiv](https://img.shields.io/badge/arXiv-2607.00736-b31b1b?style=flat-square)](https://arxiv.org/abs/2607.00736)
[![Project Page](https://img.shields.io/badge/Project-Page-4F70F2?style=flat-square)](https://startnew.github.io/projects/flexdepth/)
![Visitors](https://api.visitorbadge.io/api/visitors?path=startnew.flexdepth&label=Visitors&countColor=%23263759&style=flat-square)

---



## News

- **[2026-07]** Code is now available!
- **[2026-07]** Project page is live at [startnew.github.io/projects/flexdepth](https://startnew.github.io/projects/flexdepth/)
- **[2026-06]** Accepted by ECCV 2026

## Links

- [Paper (arXiv)](https://arxiv.org/abs/2607.00736)
- [Project Page](https://startnew.github.io/projects/flexdepth/)
- [Video Results](https://startnew.github.io/projects/flexdepth/#comparison)
- [Google Drive](https://drive.google.com/drive/folders/1sOp04-zCwkC3JJN9gMbu2GbjUdAJfp6r?usp=sharing) / [HuggingFace](https://huggingface.co/StarNew/flexdepth) /[Baidu Netdisk](https://pan.baidu.com/s/1U5vtDhDr2WH3v6L6NeNKKA?pwd=zncb).

## Table of Contents
- [Installation](#installation)
- [Prepare Datasets](#prepare-datasets)
  - [KITTI](#kitti)
  - [Cityscapes](#cityscapes)
  - [Pretrained YOLO11 Weights](#pretrained-yolo11-weights)
- [Documentation](#documentation)
  - [Training](docs/train.md)
  - [Inference](docs/inference.md)
  - [Evaluation](docs/eval.md)
  - [ONNX Export](docs/onnx.md)
- [Results](#results)
  - [Comparison with Depth Anything 2 (Eigen benchmark, Least-Squares Alignment)](#comparison-with-depth-anything-2-eigen-benchmark-least-squares-alignment)
  - [Cityscapes](#cityscapes-1)
- [Pretrained Models](#pretrained-models)
- [Citation](#citation)
- [Acknowledgment](#acknowledgment)

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

The encoder is initialised from YOLO11 segmentation weights, which must be placed in `./ckpt/`. Download
`yolo11{n,s,m,l,x}-seg.pt` from [Ultralytics v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0) and
match the file name to the scale you are training — see [Training](docs/train.md#prerequisites) for the full table.

## Documentation

| Document | Contents |
|----------|----------|
| [Training](docs/train.md) | Training commands for all five scales (Nano / Small / Medium / Large / X-Large) on KITTI and Cityscapes, plus the split stage-1 / stage-2 recipe. |
| [Inference](docs/inference.md) | Running a trained model on your own images, and reading the output. |
| [Evaluation](docs/eval.md) | Evaluation commands and the full result tables for KITTI Eigen, KITTI Eigen benchmark (DA2 comparison), and Cityscapes. |
| [ONNX Export](docs/onnx.md) | Exporting a trained model to ONNX. |

## Results

**KITTI Eigen split (Cap 80m):**

| Model | Params | GFLOPs | Abs Rel ↓ | Sq Rel ↓ | RMSE ↓ | RMSE log ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ |
|-------|--------|--------|-----------|----------|--------|------------|----------|-----------|-----------|
| Flex-Nano | 1.5M | 0.7 | 0.110 | 0.794 | 4.678 | 0.184 | 0.878 | 0.961 | 0.983 |
| Flex-Small | 6.1M | 2.8 | 0.104 | 0.713 | 4.458 | 0.179 | 0.890 | 0.964 | 0.983 |
| Flex-Medium | 12.7M | 10.0 | 0.096 | 0.639 | 4.253 | 0.172 | 0.903 | 0.968 | 0.985 |
| Flex-Large | 15.2M | 11.5 | 0.095 | 0.642 | 4.199 | 0.171 | 0.906 | 0.968 | 0.984 |
| Flex-X-Large | 32.3M | 24.6 | **0.093** | **0.605** | **4.114** | **0.167** | **0.910** | **0.969** | **0.985** |

For the evaluation commands and the full protocol, see [Evaluation](docs/eval.md).

### Comparison with Depth Anything 2 (Eigen benchmark, Least-Squares Alignment)

The improved Ground Truth  uses 5 consecutive frames with stereo completion to handle dynamic objects,covering 652 of 697 Eigen split test frames (93%) ,This split is usually called the KITTI Eigen benchmark split as monodepth2 introduce. benchmark labeled data from [official web](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) and follow [monodepth2](https://github.com/nianticlabs/monodepth2) sec. KITTI evaluation use prepare gt_depth.npz in ./splits/eigen_benchmark:

To compare with Depth Anything 2 and other zero-shot depth models, we evaluate Flex-X-Large on the KITTI Eigen benchmark split using **dense ground truth** with **least-squares alignment** (instead of the median scaling alignment used in the standard evaluation above). This aligns with the evaluation protocol used by DA2. The commands to generate the dense ground truth and run this evaluation are in [Evaluation](docs/eval.md#comparison-with-depth-anything-2-eigen-benchmark-least-squares-alignment).

**Results on KITTI Eigen benchmark (Dense GT, Least-Squares Alignment):**

| Method | Type | Params | GFLOPs | Resolution | Abs Rel ↓ | δ<1.25 ↑ |
|--------|------|--------|--------|------------|-----------|----------|
| DA2 (ViT-L) | Zero-Shot | 335M | 1947 | 1722×518 | 0.070 | **0.956** |
| DA2 (ViT-S) | Zero-Shot | 25M | 137 | 1722×518 | 0.077 | 0.944 |
| DA2 (ViT-L) | Zero-Shot | 335M | 276 | 644×196 | 0.092 | 0.915 |
| DA2 (ViT-S) | Zero-Shot | 25M | 19 | 644×196 | 0.110 | 0.881 |
| Flex-X-Large (Ours) | Self-Supervised | 32M | 25 | 640×192 | **0.063** | 0.952 |

### Cityscapes

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

## Citation

```bibtex
@InProceedings{flexdepth,
  author={Zhu, Zhaowen and Zhang, Li and Chen, Yujie and Zhang, Tian and Wang, Yingjie and Zhan, Mingxia},
  editor={Favaro, Paolo and Kukelova, Zuzana and Maki, Atsuto and Rohrbach, Anna and Schindler, Konrad and Tombari, Federico},
  title={Towards Robust Driving Perception: A Flexible Scale-Driven Family for Self-Supervised Monocular Depth Estimation},
  booktitle={Computer Vision -- ECCV 2026},
  year={2026},
  publisher={Springer Nature Switzerland},
  address={Cham},
  pages={110--129},
  isbn={978-3-032-36846-1},
  doi={10.1007/978-3-032-36846-1_7}
}
```

## Acknowledgment

This work is supported by the National Natural Science Foundation of China under Grant 62332016.

Our code is built upon [Monodepth2](https://github.com/nianticlabs/monodepth2), [Manydepth](https://github.com/nianticlabs/manydepth), and [Ultralytics](https://github.com/ultralytics/ultralytics).

We thank [DynamicDepth](https://github.com/AutoAILab/DynamicDepth) [DiPE](https://github.com/HalleyJiang/DiPE/tree/main) for providing dynamic scene annotations on the Cityscapes and KITTI test sets, respectively, which are used to evaluate model performance separately on dynamic and static regions, We also thank [DSI-training](https://github.com/zhangtian33/DSI-training) for providing the masking approach.
