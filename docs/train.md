
# Training

> ~~**Note:** Training code (`train.py`, `trainer.py`) is not included in this release but will be available soon. The commands below are provided for reference when the training code is released.~~

## Prerequisites

The encoder is initialised from YOLO11 segmentation weights, which must be present in `./ckpt/` — the path is
resolved relative to the repository root, so training has to be launched from there. Download them from
[Ultralytics v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0) and match the file name to the scale
you are training:

| Scale | Encoder | Download |
|-------|---------|----------|
| Nano | `yolo11n-seg` | [yolo11n-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt) |
| Small | `yolo11s-seg` | [yolo11s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt) |
| Medium | `yolo11m-seg` | [yolo11m-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt) |
| Large | `yolo11l-seg` | [yolo11l-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt) |
| X-Large | `yolo11x-seg` | [yolo11x-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt) |

> **Tip:** Since the YOLO encoder structure remains unchanged from YOLO11 to YOLO26, you can also use YOLO26 COCO
> segmentation pretrained weights for encoder initialization, which may yield better results. Adjust hyperparameters
> accordingly.

The `--encoder_model_type` and `--decoder_model_type` scales must match (e.g. `yolo11n-seg` with `flexn`); a mismatch
is rejected when the options are parsed.

## KITTI

> **Note:** Our reported results were obtained with N/S/M/L models trained on an RTX 2080 Ti and X-Large on an RTX 4090. Adjust `--batch_size` according to your GPU memory.



**Two-stage training in one command** (recommended):

```bash
# Flex-Nano (2080 Ti, lr=1e-4, bs=12)  -> logs/kitti_flex_n/mono_model/
python train.py --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 20 \
    --resume --scale 4 --optim NAdam --learning_rate 1e-4 \
    --encoder_model_type yolo11n-seg --decoder_model_type flexn --batch_size 12 \
    --height 192 --width 640 --dy_mu --png \
    --log_dir ./logs/kitti_flex_n --model_name mono_model

# Flex-Small (2080 Ti, lr=1e-4, bs=12)  -> logs/kitti_flex_s/mono_model/
python train.py --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 20 \
    --resume --scale 4 --optim NAdam --learning_rate 1e-4 \
    --encoder_model_type yolo11s-seg --decoder_model_type flexs --batch_size 12 \
    --height 192 --width 640 --dy_mu --png \
    --log_dir ./logs/kitti_flex_s --model_name mono_model

# Flex-Medium (2080 Ti, lr=5e-5, bs=6)  -> logs/kitti_flex_m/mono_model/
python train.py --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 20 \
    --resume --scale 4 --optim NAdam --learning_rate 5e-5 \
    --encoder_model_type yolo11m-seg --decoder_model_type flexm --batch_size 6 \
    --height 192 --width 640 --dy_mu --png \
    --log_dir ./logs/kitti_flex_m --model_name mono_model

# Flex-Large (2080 Ti, lr=5e-5, bs=6)  -> logs/kitti_flex_l/mono_model/
python train.py --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 20 \
    --resume --scale 4 --optim NAdam --learning_rate 5e-5 \
    --encoder_model_type yolo11l-seg --decoder_model_type flexl --batch_size 6 \
    --height 192 --width 640 --dy_mu --png \
    --log_dir ./logs/kitti_flex_l --model_name mono_model

# Flex-X-Large (4090, lr=5e-5, bs=12)  -> logs/kitti_flex_x/mono_model/
python train.py --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 20 \
    --resume --scale 4 --optim NAdam --learning_rate 5e-5 \
    --encoder_model_type yolo11x-seg --decoder_model_type flexx --batch_size 12 \
    --height 192 --width 640 --dy_mu --png \
    --log_dir ./logs/kitti_flex_x --model_name mono_model
```

**Or train two stages separately:**

```bash
# Stage 1 only (example: Flex-X-Large)  -> logs/kitti_flex_x/mono_model/
python train.py --num_epochs 30 --resume --scale 4 --optim NAdam \
    --learning_rate 5e-5 --encoder_model_type yolo11x-seg --decoder_model_type flexx \
    --batch_size 12 --height 192 --width 640 --dy_mu --png \
    --log_dir ./logs/kitti_flex_x --model_name mono_model

# Stage 2 only (skip stage 1)  -> same log_dir/model_name as stage 1, so --resume finds its weights
python train.py --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 20 \
    --resume --scale 4 --optim NAdam --learning_rate 5e-5 \
    --encoder_model_type yolo11x-seg --decoder_model_type flexx --batch_size 12 \
    --height 192 --width 640 --dy_mu --png --skip_step1 \
    --log_dir ./logs/kitti_flex_x --model_name mono_model
```

## Cityscapes

> **Note:** `--png` only affects KITTI; the preprocessed Cityscapes images are `.jpg`, so the flag is omitted below.

```bash
# Flex-Nano (2080ti, lr=1e-4, bs=12)  -> logs/cs_flex_n/mono_model/
python train.py --dataset cityscapes_preprocessed --split cityscapes_preprocessed \
    --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 10 \
    --resume --scale 4 --optim NAdam --learning_rate 1e-4 \
    --encoder_model_type yolo11n-seg --decoder_model_type flexn --batch_size 12 \
    --height 192 --width 512 --dy_mu --data_path <cityscapes_path> \
    --log_dir ./logs/cs_flex_n --model_name mono_model

# Flex-Small (2080ti, lr=1e-4, bs=12)  -> logs/cs_flex_s/mono_model/
python train.py --dataset cityscapes_preprocessed --split cityscapes_preprocessed \
    --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 10 \
    --resume --scale 4 --optim NAdam --learning_rate 1e-4 \
    --encoder_model_type yolo11s-seg --decoder_model_type flexs --batch_size 12 \
    --height 192 --width 512 --dy_mu --data_path <cityscapes_path> \
    --log_dir ./logs/cs_flex_s --model_name mono_model

# Flex-Medium (2080ti, lr=1e-4, bs=6)  -> logs/cs_flex_m/mono_model/
python train.py --dataset cityscapes_preprocessed --split cityscapes_preprocessed \
    --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 10 \
    --resume --scale 4 --optim NAdam --learning_rate 1e-4 \
    --encoder_model_type yolo11m-seg --decoder_model_type flexm --batch_size 6 \
    --height 192 --width 512 --dy_mu --data_path <cityscapes_path> \
    --log_dir ./logs/cs_flex_m --model_name mono_model

# Flex-Large (2080ti, lr=5e-5, bs=6)  -> logs/cs_flex_l/mono_model/
python train.py --dataset cityscapes_preprocessed --split cityscapes_preprocessed \
    --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 10 \
    --resume --scale 4 --optim NAdam --learning_rate 5e-5 \
    --encoder_model_type yolo11l-seg --decoder_model_type flexl --batch_size 6 \
    --height 192 --width 512 --dy_mu --data_path <cityscapes_path> \
    --log_dir ./logs/cs_flex_l --model_name mono_model

# Flex-X-Large  -> logs/cs_flex_x/mono_model/
python train.py --dataset cityscapes_preprocessed --split cityscapes_preprocessed \
    --use_var_net --use_step_2 --num_epochs 30 --start_opt_epoch 29 --step_2_epoch 10 \
    --resume --scale 4 --optim NAdam --learning_rate 5e-5 \
    --encoder_model_type yolo11x-seg --decoder_model_type flexx --batch_size 6 \
    --height 192 --width 512 --dy_mu --data_path <cityscapes_path> \
    --log_dir ./logs/cs_flex_x --model_name mono_model
```

## See also

- [Inference](inference.md) — run a trained checkpoint on your own images
- [Evaluation](eval.md) — evaluate the checkpoints produced above
- [ONNX Export](onnx.md) — export a trained model to ONNX
