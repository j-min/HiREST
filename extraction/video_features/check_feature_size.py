import torch
from glob import glob
from tqdm import tqdm
import json
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_file", default="./data/splits/all_data_test.json", type=str)
parser.add_argument("--feature_folder",
                    default="./data/eva_clip_features/", type=str)

args = parser.parse_args()

durations = {}

with open(args.data_file, 'r') as f:
    prompt2video_anns = json.load(f)
    for i, (prompt, video_anns) in enumerate(prompt2video_anns.items()):
        has_relevant_videos = False
        for video_fname, video_ann in video_anns.items():
            video_duration = video_ann['v_duration']
            video_duration = round(video_duration)
            durations[video_fname] = video_duration


feature_files = glob(f"{args.feature_folder}*.pt")
for f in tqdm(feature_files):
    features = torch.load(f, map_location='cpu')

    name = f.split("/")[-1].replace(".pt", "")

    if features.shape[0] != durations[name]:
        features = features[0:durations[name]]
        torch.save(feature_files, f)
