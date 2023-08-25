import argparse
import json
import os
from glob import glob
from tqdm import tqdm
from pytube import YouTube

BASE_URL = "https://www.youtube.com/watch?v="

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='./data/splits/')
    parser.add_argument('--save_path', type=str, default='./data/videos/')
    args = parser.parse_args()

    data_folder = args.data_folder
    save_path = args.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    video_ids = [ ]

    for file in glob(f"{data_folder}/*.json"):
        with open(file, 'r') as f:
            data = json.load(f)
        
        for prompt in data:
            videos = list(data[prompt].keys())
            video_ids.extend([BASE_URL + video.replace(".mp4", "") for video in videos])

    for v in tqdm(video_ids):
        try:
            yt = YouTube(v) 
        except: 
            print("Connection Error")
    
        try:
            yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(output_path=save_path, filename=f'{v.split("=")[-1]}.mp4')
        except:
            print("Error downloading video")