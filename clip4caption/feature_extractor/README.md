# How to extract the video features from CLIP4Clip
## Prepare the Pretrained CLIP4Clip Model
Train CLIP4Clip model using https://github.com/ArrowLuo/CLIP4Clip , and put the pretrained model with the name of `pytorch_model.bin` in the `pretrained_clip4clip/msvd` or `pretrained_clip4clip/msrvtt` folder. Or you can download the pretrained model of CLIP4Clip (ViT-B/32) from this [link](https://drive.google.com/drive/folders/16BlJLtGMrGmIty56wAO4POh-1Uw_AJ8o?usp=sharing).

## Prepare the Dataset
Dowload the raw videos of MSVD ([link](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/)) and MSR-VTT ([link](https://github.com/VisionLearningGroup/caption-guided-saliency/issues/6)), and put the videos in `raw` folder as displayed below
```bash
├── feature_extractor
├── dataset
│   ├── MSVD
│   │   ├── raw # put the 1970 raw videos in here
│   │   ├── captions 
│   │   ├── raw-captions_mapped.pkl # mapping between video id with captions
│   │   ├── train_list_mapping.txt
│   │   ├── val_list_mapping.txt
│   │   ├── test_list_mapping.txt
│   ├── MSRVTT
│   │   ├── raw # put the 10000 raw videos in here
│   │   ├── msrvtt.csv # list of video id in msrvtt dataset
│   │   ├── MSRVTT_data.json # metadata of msrvtt dataset, which includes video url, video id, and caption
```
## Extract the Features
Execute the following command to extract the video features and to save the features at `../extracted_feats` folder:
### MSR-VTT
```bash
python clip_feature_extractor.py --dataset_type msrvtt --save_dir ../extracted_feats --dataset_dir ../dataset
```
### MSVD
```bash
python clip_feature_extractor.py --dataset_type msvd --save_dir ../extracted_feats --dataset_dir ../dataset
```

Note that you may need to modify the arguments as per your needs.
