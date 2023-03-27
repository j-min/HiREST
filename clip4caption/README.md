# Reproducing CLIP4Caption
This is the first unofficial implementation of CLIP4Caption method (ACMMM 2021), which is the SOTA method in video captioning task at the time when this project was implemented.

![image](https://user-images.githubusercontent.com/5786636/210189414-aef876e0-38ab-4026-8ece-4a1f803a1005.png)

**Note**: The provided extracted features and the reproduced results are not obtained using TSN sampling as in the CLIP4Caption paper. However, even without TSN sampling, i.e., only using original sampling method in CLIP4Clip, it is found that the similar (even slightly better) performance results can be achieved as in the CLIP4Caption paper. While reproducing the results, it was observed that using TSN sampling could not achieve the similar performance results as in the paper.

**Paper**: Mingkang Tang, Zhanyu Wang, Zhenhua LIU, Fengyun Rao, Dian Li, and Xiu Li. 2021. CLIP4Caption: CLIP for Video Caption. In Proceedings of the 29th ACM International Conference on Multimedia (MM '21). Association for Computing Machinery, New York, NY, USA, 4858–4862. > https://dl.acm.org/doi/10.1145/3474085.3479207

## Reproduced Results
### MSRVTT
| Method  | BLEU@4 | METEOR | ROUGE-L | CIDEr | Checkpoint |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| CLIP4Caption* (ViT-B/32) | 46.1  | 30.7  | 63.7  | 57.7  | - |
| CLIP4Caption (ViT-B/32)  | 46.41  | 30.72  | 63.90 | 58.42 | [link](https://drive.google.com/file/d/17p476sL5_KZoQ2h4e1TU7-qWH4VqzReT/view?usp=sharing) |
| CLIP4Caption (ViT-B/16)  | 48.37 | 31.53  | 65.20 | 61.34 | [link](https://drive.google.com/file/d/1WyLktrWLrHMR_ymt_yQHWd4zyvnWSc71/view?usp=sharing) |

(*) Original results from the paper

## Setup
Execute below scripts in the main folder, to avoid download conflict when doing distributed pretrain.

```bash
mkdir modules/bert-base-uncased
cd modules/bert-base-uncased/
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
mv bert-base-uncased-vocab.txt vocab.txt
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz
tar -xvf bert-base-uncased.tar.gz
rm bert-base-uncased.tar.gz
cd ../../
```

Prepare the conda environment:
```bash
conda create -n clip4caption python=3.6.9 tqdm boto3 requests pandas
conda activate clip4caption
pip install torch==1.10.2 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install git+https://github.com/Maluuba/nlg-eval.git@master
pip install pycocoevalcap
pip install pickle5
pip install opencv-python==4.5.5.62
```

Download the pretrained weight of UniVL:
```bash
mkdir -p ./weight
wget -P ./weight https://github.com/microsoft/UniVL/releases/download/v0/univl.pretrained.bin
```

## Extract the Video Features
The extracted features (ViT-B/32) are provided [here](https://drive.google.com/drive/folders/1GHpAKDNU3qZxzIk6zqdatMjQoGc6Q3_9?usp=sharing). However if you want to extract the features by yourself, follow the instructions written [here](https://github.com/willyfh/clip4caption/tree/main/feature_extractor)

## Training & Evaluation
The shell scripts to train and to evaluate the model is provided [here](https://github.com/willyfh/clip4caption/tree/main/scripts). You may need to modify the scripts as per your needs.

## References
This repository is implemented based on [UniVL](https://github.com/microsoft/UniVL) and [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip)
