import cv2
import os
import json
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
import argparse
from glob import glob

def extract_frames_method2(video_path):
    video_path = video_path.replace("\n", "")
    video_fname = Path(video_path).name
    x = str(VIDEOS_DIR / video_path)
    
    vidcap = cv2.VideoCapture(x)

    frame = 0
    
    success = True
    while success:
        curr_frame_str = str(frame).zfill(6)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, frame*1000)
        success,image = vidcap.read()

        if not success:
            break

        if frame % RATE == 0:
            frame_save_path = FRAMES_SAVE_DIR / video_fname / f'frame_{curr_frame_str}.jpg'
            if not os.path.exists(frame_save_path.parent):
                os.makedirs(frame_save_path.parent)

            frame_save_path = str(frame_save_path)
            cv2.imwrite(frame_save_path, image)

        frame += 1

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)

    args = parser.parse_args()
    print(args)
    print('CV2 version:', cv2.__version__)

    with open(args.data_path, 'r') as f:
        prompt2video_anns = json.load(f)
    print('Loaded data from', args.data_path)
    
    FRAMES_SAVE_DIR = Path(args.save_path)
    VIDEOS_DIR = Path(args.video_path)
    RATE = 1/1

    if not os.path.exists(FRAMES_SAVE_DIR):
        os.makedirs(FRAMES_SAVE_DIR)
    print('Saving frames to', FRAMES_SAVE_DIR)

    all_video_paths = []

    for prompt in tqdm(prompt2video_anns):
        for video_fname in prompt2video_anns[prompt]:
            if video_fname not in all_video_paths:
                if os.path.exists(f"{VIDEOS_DIR}/{video_fname}"):
                    all_video_paths.append(video_fname)
                        
    print('Extracting frames from', len(all_video_paths), 'videos')

    if args.workers > 1:
        print('Using', args.workers, 'workers')
        with Pool(args.workers) as pool:
            all_results = list(tqdm(pool.imap(extract_frames_method2, all_video_paths), total=len(all_video_paths)))
    else:
        print('Using 1 worker')
        all_results = [extract_frames_method2(video_path) for video_path in tqdm(all_video_paths)]

    print('Done')