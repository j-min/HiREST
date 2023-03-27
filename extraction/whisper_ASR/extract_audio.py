from pathlib import Path
from tqdm import tqdm
import subprocess
import multiprocessing
import json
import argparse

def extract_audio(video_path):
    out_fname = out_dir / (video_path.stem + '.wav')

    cmd = ['ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', '-f', 'wav',
            '-loglevel', 'warning', '-hide_banner', '-stats',
            str(out_fname)]
    subprocess.call(cmd)

if __name__ == '__main__':
    print('Extracting audio from videos...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)

    args = parser.parse_args()

    video_dir = Path(args.video_path)
    out_dir = Path(args.save_path)

    prompt2video_anns = json.load(open(args.data_path))

    video_fnames = []
    for i, (prompt, video_anns) in enumerate(prompt2video_anns.items()):
        for video_fname, video_ann in video_anns.items():
            video_fnames.append(video_fname)

    video_fnames = list(set(video_fnames))

    video_paths = [video_dir / video_fname for video_fname in video_fnames]

    print(f'Found {len(video_paths)} videos at {video_dir}')

    print('Will save audio to', out_dir)

    print(f'Multiprocessing with {args.workers} workers')
    with multiprocessing.Pool(args.workers) as pool:
        list(tqdm(pool.imap(extract_audio, video_paths), total=len(video_paths)))
