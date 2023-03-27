
import os
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import clip

class VideoFramesDataset(Dataset):
    def __init__(self, frame_dir, video_ids, preprocess_fn, args):
        """
        Dataset to extract features directly from video frames.
        Used to batch video feature extraction.
        """

        self.all_frame_dir = Path(frame_dir)
        self.video_ids = video_ids
        self.args = args
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):

        video_id = self.video_ids[idx]

        video_frame_dir = self.all_frame_dir / video_id
        
        # frame_000000.jpg to frame_000031.jpg
        frame_paths = [str(video_frame_dir / f"frame_{str(i).zfill(6)}.jpg") for i in range(args.n_model_frames)]

        if self.args.n_model_frames > 0:
            n_frames = len(frame_paths)
            # Uniformly subsample via linspace
            frame_ids = np.linspace(0, n_frames - 1, self.args.n_model_frames).astype(int)
            frame_paths = [frame_paths[i] for i in frame_ids]

        frames = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert('RGB')
            frame = self.preprocess_fn(img)
            frames.append(frame)

        frames = torch.stack(frames)
        # assert frames.shape == (32, 3, 224, 224)

        return frames
    
    def collate_fn(self, batch):
        batch_frames = torch.stack(batch)
        # assert batch_frames.shape == (self.args.batch_size, 32, 3, 224, 224)
        return batch_frames
    
    def get_dataloader(self, batch_size=10, num_workers=4):
        dataloader = DataLoader(self,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                collate_fn=self.collate_fn)
        return dataloader


class VideoRetrievalDataset(Dataset):
    def __init__(self, split, args):
        """
        Dataset to for video retrieval.
        only used for dataset creation.
        not used with dataloader.
        """
        self.args = args

        self.data = []
        self.prompts = []
        self.videos = []

        self.video_durations = []

        self.video_feat_dir = args.video_feature_dir
        self.asr_dir = args.asr_dir

        with open(f"{args.data_dir}/all_data_{split}.json", 'r') as f:
            data = json.load(f)

            for prompt in data:
                self.prompts.append(prompt)

                for video in data[prompt]:
                    self.videos.append(video)

                    self.data.append({
                        "video_id": video.replace(".mp4", ""),
                        "clip_feature": f"{self.video_feat_dir}/{video}.pt",
                        "asr": f"{self.asr_dir}/{video.replace('.mp4', '')}.srt",
                        "target": prompt,
                        "v_duration": data[prompt][video]["v_duration"]
                    })

                    self.video_durations.append(data[prompt][video]["v_duration"])

        print(f"self.videos: {len(self.videos)}")
        print(f"self.prompts: {len(self.prompts)}")


class NegativeVideoRetrievalDataset(Dataset):
    def __init__(self, split, args):
        """
        Dataset to include negative distractors for video retrieval.
        only used for dataset creation.
        not used with dataloader.
        """
        self.args = args

        self.data = []
        self.prompts = []
        self.videos = []

        self.video_durations = []

        self.video_feat_dir = args.video_feature_dir
        self.asr_dir = args.asr_dir

        print(f'split: {split}')

        with open(f"{args.data_dir}/all_data_{split}.json", 'r') as f:
            data = json.load(f)

            for prompt in data:
                self.prompts.append(prompt)

                for video in data[prompt]:
                    self.videos.append(video)

                    self.data.append({
                        "video_id": video.replace(".mp4", ""),
                        "clip_feature": f"{self.video_feat_dir}/{video}.pt",
                        "asr": f"{self.asr_dir}/{video.replace('.mp4', '')}.srt",
                    })

        print(f"self.videos: {len(self.videos)}")
        print(f"self.prompts: {len(self.prompts)}")



if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    args = parser.parse_args()

    from accelerate.utils import set_seed
    import random
    import numpy as np

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = args.device

    if args.video_retrieval_model == 'clip':
        clip_model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
        if args.load is not None:
            LOAD = args.load
            print("Loaded from:", LOAD)
            clip_model.load_state_dict(torch.load(LOAD, map_location='cpu'))
        clip_model = clip_model.to(device)
        clip_model.eval()

    elif args.video_retrieval_model == 'clip_g':
        import sys
        sys.path.append("./EVA_clip")
        from eva_clip import build_eva_model_and_transforms
        clip_model, clip_preprocess = build_eva_model_and_transforms(
            "EVA_CLIP_g_14",
            pretrained='./pretrained_weights/eva_clip_psz14.pt')
        print("Loaded EVA CLIP G")
        clip_model = clip_model.to(device)
        clip_model.eval()

    test_dataset = VideoRetrievalDataset("test", args)

    prompts = test_dataset.prompts
    all_video_ids = test_dataset.videos

    # Loading distractor videos to augment the video in the test set
    distractor_dataset = NegativeVideoRetrievalDataset("test_negative_samples", args)
    all_video_ids = test_dataset.videos + distractor_dataset.videos

    print('Number of prompts: ', len(prompts))
    print('Number of videos: ', len(all_video_ids))

    batch_size = args.eval_batch_size
    print("Computing text embeddings")
    
    all_text_embeds = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Computing text embeddings", colour="green"):
        with torch.no_grad():
            if args.video_retrieval_model in ['clip', 'clip_g']:
                text_tokens = clip.tokenize(prompts[i:i+batch_size]).to(device)
                text_embeds = clip_model.encode_text(text_tokens)
                
            text_embeds = text_embeds.float()
            text_embeds = text_embeds.to("cpu")
            text_embeds /= text_embeds.norm(dim=-1, keepdim=True)            
            
        all_text_embeds.append(text_embeds)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    print(f"Text embeddings shape: {all_text_embeds.shape}")

    print("Computing video embeddings")

    if args.raw_frame:
        video_frame_dir = args.video_dir
        # frame_dir, video_ids, preprocess_fn, args):
        print("Using raw frames")
        print("Video frame dir: ", video_frame_dir)

        if args.save_feats:
            num_process = 1
            process_id = 0
            if args.num_process > 0:
                num_process = args.num_process
                process_id = args.process_id

                print("All video ids: ", len(all_video_ids))
                print("Num process: ", num_process)
                all_video_ids = all_video_ids[process_id::num_process]
                print("Video ids: ", len(all_video_ids))
                print("Process id: ", process_id)

        video_frame_dataset = VideoFramesDataset(
            video_frame_dir,
            all_video_ids,
            clip_preprocess,
            args,
            )
        
        video_frame_dataloader = video_frame_dataset.get_dataloader(
            batch_size=batch_size,
            num_workers=args.num_workers,
        )

        all_video_embeds = []

        if args.save_feats:
            os.makedirs(args.video_feature_dir, exist_ok=True)
            print("Saving feats to: ", args.video_feature_dir)

        for i, batch in enumerate(tqdm(video_frame_dataloader, desc=f"Computing video embeddings - N frames: {args.n_model_frames}", colour="green")):

            B = batch.shape[0]
            frames = batch
            assert frames.shape == (B, args.n_model_frames, 3, 224, 224), f"Batch shape: {frames.shape}"

            frames = frames.to(device)

            if args.video_retrieval_model in ['clip', 'clip_g']:

                frames = frames.view(-1, 3, 224, 224)

                with torch.no_grad():
                    video_embeds = clip_model.encode_image(frames)
                    video_embeds = video_embeds.float()

                    video_embeds = video_embeds.view(B, args.n_model_frames, 1024)

                if args.save_feats:
                    for j in range(B):
                        video_id = all_video_ids[i*batch_size + j]
                        video_feat_dir = Path(args.video_feature_dir)
                        video_feat_path = video_feat_dir / f"{video_id}.pt"
                        torch.save(video_embeds[j], video_feat_path)

                # Avgpool
                video_embeds = video_embeds.mean(dim=1, keepdim=False)
                
                video_embeds /= video_embeds.norm(dim=-1, keepdim=True)
                video_embeds = video_embeds.to("cpu")

            all_video_embeds.append(video_embeds)

    else:

        all_video_embeds = []
        video_feat_dir = Path(args.video_feature_dir)
        print(f"Video feature dir: {video_feat_dir}")
        print(f"Video feature dir exists: {video_feat_dir.exists()}")
        print(f"N frames: {args.n_model_frames}")

        for i in tqdm(range(len(all_video_ids)), desc=f"Computing video embeddings - N frames: {args.n_model_frames}", colour="green"):

            video_id = all_video_ids[i]
            # video_duration = all_video_durations[i]

            video_feat_dir = Path(args.video_feature_dir)
            video_feat_path = video_feat_dir / f"{video_id}.pt"

            video_embeds = torch.load(video_feat_path, map_location="cpu")

            video_duration = video_embeds.shape[0]

            if args.n_model_frames > 0:
                # video_features: [n_frames, 512]
                n_frames = video_embeds.shape[0]
                # Uniformly subsample via linspace
                # if n_frames > args.n_model_frames:
                frame_ids = np.linspace(0, n_frames - 1, args.n_model_frames).astype(int)
                frame_ids = torch.from_numpy(frame_ids)
                video_embeds = video_embeds[frame_ids]
            
            video_embeds = video_embeds.float()

            if args.video_retrieval_model in ['clip', 'clip_g']:
                # CLIP-zeroshot avgpool
                video_embeds = video_embeds.mean(dim=0, keepdim=True)
            
            video_embeds = video_embeds.to("cpu")
            video_embeds /= video_embeds.norm(dim=-1, keepdim=True)
            all_video_embeds.append(video_embeds)
        
        
    all_video_embeds = torch.cat(all_video_embeds, dim=0)
    print(f"Video embeddings shape: {all_video_embeds.shape}")

    print("Computing scores")
    text_to_video_scores = torch.matmul(all_text_embeds, all_video_embeds.T)
    print(f"Scores shape: {text_to_video_scores.shape}")

    prompt_video_scores = { }
    for i, prompt in enumerate(tqdm(prompts, desc="Preparing output json", colour="green")):
        prompt_video_scores[prompt] = {
            "videos": [],
            "scores": []
        }

        prompt_video_scores[prompt]["videos"] = all_video_ids
        prompt_video_scores[prompt]["scores"] = text_to_video_scores[i].tolist()

    save_dir = Path("VR_results")
    if not save_dir.exists():
        save_dir.mkdir()

    # "clip_FT_avgpool.json"
    save_path = save_dir / f"{args.run_name}.json"
    with open(save_path, 'w') as f:
        json.dump(prompt_video_scores, f, indent=4)
    print(f"Saved results to {save_path}")
            

    