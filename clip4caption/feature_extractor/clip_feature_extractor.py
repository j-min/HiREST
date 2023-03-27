import argparse
import os
import cv2
import numpy as np
from numpy import dot
from numpy.linalg import norm
import sys
import glob
import json
import math
from tqdm import tqdm
import torch

from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import pickle
import pathlib

from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.optimization import BertAdam
from util import parallel_apply, get_logger

sys.path.append("../dataloaders/")
from dataloader_msvd_raw import MSVD_Raw_DataLoader
from dataloader_msrvtt_raw import MSRVTT_Raw_DataLoader



# Argument
class args:
    msvd = True # or msvd = False for MSR-VTT
    max_frames = 20
    pretrined_clip4clip_dir='pretrained'
    
def get_args():
    parser = argparse.ArgumentParser(description="CLIP Feature Extractor")
    parser.add_argument('--dataset_type', choices=['msvd', 'msrvtt'], default='msvd', type=str, help='msvd or msrvtt')
    parser.add_argument('--dataset_dir', type=str, default='../dataset', help='should be pointed to the location where the MSVD and MSRVTT dataset located')
    parser.add_argument('--save_dir', type=str, default='../extracted_feats', help='location of the extracted features')
    parser.add_argument('--slice_framepos', choices=[0,1,2], type=int, default=2,
                        help='0: sample from the first frames; 1: sample from the last frames; 2: sample uniformly.')
    parser.add_argument('--max_frames', type=int, default=20, help='max sampled frames')
    parser.add_argument('--frame_order', type=int, choices=[0,1,2], default=0, help='0: normal order; 1: reverse order; 2: random order.')
    parser.add_argument('--pretrained_clip4clip_dir', type=str, default='pretrained_clip4clip/', help='path to the pretrained CLIP4Clip model') 
    parser.add_argument('--device', choices=["cpu", "cuda"], type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--pretrained_clip_name', type=str, choices=["ViT-B/32", "ViT-B/16"], default="ViT-B/32")
    
    args = parser.parse_args()
    
    if args.device == "cuda":
        args.device = torch.device('cuda')
    
    if args.dataset_type=="msvd":
        dset_path = os.path.join(args.dataset_dir,'MSVD')
        args.videos_path = os.path.join(dset_path,'raw') # video .avi    

        args.data_path =os.path.join(os.path.join(dset_path,'captions','youtube_mapping.txt'))
        args.max_words = 30
        
        save_dir = os.path.join(args.save_dir, "msvd")
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        args.save_file = os.path.join(save_dir,'MSVD_CLIP4Clip_features.pickle')
        
        args.pretrained_clip4clip_path = os.path.join(args.pretrained_clip4clip_dir, 'msvd','pytorch_model.bin')

    elif args.dataset_type=="msrvtt":
        dset_path = os.path.join(args.dataset_dir,'MSRVTT')
        args.videos_path = os.path.join(dset_path,'raw') 
        
        args.data_path=os.path.join(dset_path,'MSRVTT_data.json')
        args.max_words = 73
        args.csv_path = os.path.join(dset_path,'msrvtt.csv')

        save_dir = os.path.join(args.save_dir, "msrvtt")
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        args.save_file = os.path.join(save_dir,'MSRVTT_CLIP4Clip_features.pickle')
        
        args.pretrained_clip4clip_path = os.path.join(args.pretrained_clip4clip_dir, 'msrvtt','pytorch_model.bin')
        
    return args
    
def get_dataloader(args):
    
    dataloader = None
    if args.dataset_type=="msvd":
        dataloader = MSVD_Raw_DataLoader(
            data_path=args.data_path,
            videos_path=args.videos_path,
            max_frames=args.max_frames,
            frame_order=args.frame_order,
            slice_framepos=args.slice_framepos,
            transform_type = 0,
        ) 
    elif args.dataset_type=="msrvtt":
        dataloader = MSRVTT_Raw_DataLoader(
            csv_path=args.csv_path,
            videos_path=args.videos_path,
            max_frames=args.max_frames,
            frame_order=args.frame_order,
            slice_framepos=args.slice_framepos,
            transform_type = 0,
        )
    return dataloader
    
def load_model(args):
    model_state_dict = torch.load(args.pretrained_clip4clip_path, map_location='cpu')
    cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained('cross-base', cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
    clip = model.clip.to(args.device)
    return clip


def main():
    args = get_args()
    dataloader = get_dataloader(args)
    model = load_model(args)
    model.eval()

    with torch.no_grad():
        data ={}
        stop = False
        with open(args.save_file, 'wb') as handle:

            for i in tqdm(range(len(dataloader))):
                video_id,video,video_mask = dataloader[i]

                tensor = video[0]
                tensor = tensor[video_mask[0]==1,:]
                tensor = torch.as_tensor(tensor).float()
                video_frame,num,channel,h,w = tensor.shape
                tensor = tensor.view(video_frame*num, channel, h, w)

                video_frame,channel,h,w = tensor.shape

                output = model.encode_image(tensor.to(args.device), video_frame=video_frame).float().to(args.device)
                output = output.detach().cpu().numpy()
                data[video_id]=output

                del output
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()