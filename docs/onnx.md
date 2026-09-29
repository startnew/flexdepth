# ONNX Export

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
