# Inference

`inference.py` runs a trained model on your own images. It requires pretrained weights placed in `./models/` —
see [Pretrained Models](../README.md#pretrained-models).

## Single image

```bash
python inference.py \
    --load_weights_folder ./models/kitti/flex_x \
    --encoder_model_type yolo11x-seg --decoder_model_type flexx \
    --image_path path/to/image.png
```

## A folder of images

```bash
python inference.py \
    --load_weights_folder ./models/kitti/flex_x \
    --encoder_model_type yolo11x-seg --decoder_model_type flexx \
    --image_path path/to/folder
```

Every `*.png` / `*.jpg` / `*.jpeg` in the folder is processed, in sorted filename order.

## Other scales

Substitute the encoder/decoder pair to match the checkpoint — the two scales must agree:

| Scale | `--encoder_model_type` | `--decoder_model_type` | `--load_weights_folder` |
|-------|------------------------|------------------------|--------------------------|
| Nano | `yolo11n-seg` | `flexn` | `./models/kitti/flex_n` |
| Small | `yolo11s-seg` | `flexs` | `./models/kitti/flex_s` |
| Medium | `yolo11m-seg` | `flexm` | `./models/kitti/flex_m` |
| Large | `yolo11l-seg` | `flexl` | `./models/kitti/flex_l` |
| X-Large | `yolo11x-seg` | `flexx` | `./models/kitti/flex_x` |

Cityscapes checkpoints in `./models/cs/` use the same flags, only the weights folder differs.

## Output

Results are written to `<image_dir>/depth/` — the folder you passed to `--image_path`, or the directory holding
the image when a single file is given. Use `--save_path` to write somewhere else.

| File | Contents |
|------|----------|
| `<name>_depth.png` | Colour depth map, sized to the input image. |
| `<name>_disp.npy` | Raw disparity, float32. Only with `--save_npy`. |
| `<name>_depth.npy` | Depth in metres, float32. Only with `--save_npy`. |

Add `--save_npy` to also dump the numeric maps:

```bash
python inference.py \
    --load_weights_folder ./models/kitti/flex_x \
    --encoder_model_type yolo11x-seg --decoder_model_type flexx \
    --image_path path/to/image.png --save_npy
```

## Scale ambiguity

The models are self-supervised and single-image, so predicted depth is only defined up to a global scale — the
metre values in `<name>_depth.npy` are not physically calibrated. Evaluation handles this with median scaling
(`evaluate_depth.py`) or least-squares alignment (`--use_lstsq_alignment`) against ground truth; there is no
ground truth here, so treat the numbers as relative. The colour map is unaffected and is the intended output for
most uses.

Depth is clipped to `--min_depth` / `--max_depth` (default `0.1` – `100.0`); pass them explicitly to match the
range your checkpoint was trained on, e.g. `--min_depth 0.1 --max_depth 80` for the KITTI models.

## See also

- [Evaluation](eval.md) — benchmark the checkpoints on KITTI / Cityscapes test splits
- [ONNX Export](onnx.md) — export a model for deployment
