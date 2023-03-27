# Video Feature Extraction

## Download Videos
```bash
python ./extraction/video_features/download_videos.py --save_path ./data/videos/
```

## Frame Extraction
```bash
python ./extraction/video_features/extract_frames.py --workers 32 --data_path ./data/splits/all_data_train.json --save_path ./data/raw_frames/ --video_path ./data/videos/
python ./extraction/video_features/extract_frames.py --workers 32 --data_path ./data/splits/all_data_val.json --save_path ./data/raw_frames/ --video_path ./data/videos/
python ./extraction/video_features/extract_frames.py --workers 32 --data_path ./data/splits/all_data_test.json --save_path ./data/raw_frames/ --video_path ./data/videos/
```

## Feature Extraction
```bash
python ./extraction/video_features/extract_features.py --frame_path ./data/raw_frames/ --save_path ./data/eva_clip_features/
```

You can optionally change batch size with `--batch_size`.
If you want to split the dataset across multiple gpus, you can use `--slice_start` and `--slice_end` to specify which video indices that the process should use and then `--device` to set the GPU. Then you can just run the command again with different start/end indices and device.

## Feature Correction
Due to some rounding differences sometimes features might be 1 frame too large. If you get an size mismatch when running the model, you can try running this script to fix it.

```bash
python ./extraction/video_features/check_feature_size.py --data_file ./data/splits/all_data_train.json --feature_folder ./data/eva_clip_features/
```