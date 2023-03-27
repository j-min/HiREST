import clip
import torch
from glob import glob
from tqdm import tqdm
from PIL import Image
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--slice_start", type=int, default=0)
parser.add_argument("--slice_end", type=int, default=-1)
parser.add_argument("--frame_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=256)

args = parser.parse_args()
print(args)

sys.path.append("./EVA_clip")
from eva_clip import build_eva_model_and_transforms

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
model, preprocess = build_eva_model_and_transforms("EVA_CLIP_g_14", pretrained="./pretrained_weights/eva_clip_psz14.pt")
model = model.to(device)

videos = glob(f"{args.frame_path}/*/")

if args.slice_end != -1:
    videos = videos[args.slice_start:args.slice_end]
else:
    videos = videos[args.slice_start:]

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


BATCH_SIZE = args.batch_size

for video in tqdm(videos, colour="green"):
    images = [ ]
    
    raw_images: list = glob(f"{video}/*.jpg")
    raw_images.sort(key=lambda x: int(x.split("/")[-1].replace(".jpg", "").split("_")[-1]))

    for image in raw_images:
        x = preprocess(Image.open(image)).cpu()
        images.append(x)

    images = torch.stack(images)

    chunks = list(divide_chunks(images, BATCH_SIZE))

    video_features = [ ]

    for chunk in chunks:
        with torch.no_grad():
            f = model.encode_image(chunk.to(device)).cpu()
        video_features.append(f)
    
    video_features = torch.cat(video_features)
    video_features /= video_features.norm(dim=-1, keepdim=True)

    # print(video)
    video_name = video.split("/")[-2]

    torch.save(video_features, f"{args.save_path}/{video_name}.pt")