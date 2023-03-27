from pathlib import Path
import argparse
from tqdm import tqdm
import srt
import torch
from sentence_transformers import SentenceTransformer
import clip

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
        choices=['sentence-transformers/all-MiniLM-L6-v2', 'clip'],
    )
    parser.add_argument("--asr_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()
    print(args)

    ASR_dir = Path(args.asr_dir)

    if 'sentence-transformers' in args.model:
        model = SentenceTransformer(args.model)
    elif 'clip' in args.model:
        model, preprocess = clip.load('ViT-B/32', device='cpu')
    
    ASR_feats_dir = Path(f'{args.save_dir}')

    device = 'cuda'

    model = model.eval()
    model = model.to(device)

    print('model loaded: ', args.model)
    print('ASR_feats_dir: ', ASR_feats_dir)

    all_ASR_paths = list(ASR_dir.glob('*.srt'))

    ASR_paths = all_ASR_paths

    for asr_path in tqdm(ASR_paths):

        with open(asr_path, 'r') as f:
            transcript_srt_str = f.read()

        all_subs = []
        for sub in srt.parse(transcript_srt_str):
            all_subs.append(sub.content)

        with torch.no_grad():
            if 'sentence-transformers' in args.model:
                sub_embeddings = model.encode(all_subs, convert_to_tensor=True)
            else:
                sub_embeddings = model.encode_text(clip.tokenize(all_subs, truncate=True).to(device)).float()

        feat_path = ASR_feats_dir / (asr_path.stem + '.pt')
        torch.save(sub_embeddings.cpu(), feat_path)