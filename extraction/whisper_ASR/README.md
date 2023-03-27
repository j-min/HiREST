# Extract wav from mp4 videos
```bash
python ./extraction/whisper_ASR/extract_audio.py --workers 32 --data_path ./data/splits/all_data_train.json --save_path ./data/audio/ --video_path ./data/videos/
python ./extraction/whisper_ASR/extract_audio.py --workers 32 --data_path ./data/splits/all_data_val.json --save_path ./data/audio/ --video_path ./data/videos/
python ./extraction/whisper_ASR/extract_audio.py --workers 32 --data_path ./data/splits/all_data_test.json --save_path ./data/audio/ --video_path ./data/videos/
```

# Extract ASR and save as SRT
```bash
python ./extraction/whisper_ASR/extract_ASR.py --model small.en --audio_dir ./data/audio/ --asr_dir ./data/ASR/
```

# Extract ASR feature
```bash
# minilm
python ./extraction/whisper_ASR/extract_ASR_embedding.py --model sentence-transformers/all-MiniLM-L6-v2 --asr_dir ./data/ASR/

# clip ViT-B/32
python ./extraction/whisper_ASR/extract_ASR_embedding.py --model clip --asr_dir ./data/ASR/
```