import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import srt
import random
import clip


def timestamp_to_frame_index(timestamp, video_duration, n_frames=32):
    """
    Convert a timestamp in seconds to a frame index.
    1) Create bins by dividing the video duration into n_frames bins.
    2) Find the bin that contains the timestamp.
    """
    video_duration = int(video_duration)
    if n_frames < 0:
        n_frames = video_duration

    bins = np.linspace(0, video_duration-1, n_frames)

    # e.g., video_duration = 200, n_frames: 32
    # bins
    # [  0.           6.41935484  12.83870968  19.25806452  25.67741935
    #   32.09677419  38.51612903  44.93548387  51.35483871  57.77419355
    #   64.19354839  70.61290323  77.03225806  83.4516129   89.87096774
    #   96.29032258 102.70967742 109.12903226 115.5483871  121.96774194
    #  128.38709677 134.80645161 141.22580645 147.64516129 154.06451613
    #  160.48387097 166.90322581 173.32258065 179.74193548 186.16129032
    #  192.58064516 199.        ]

    bin_index = np.digitize(timestamp, bins, right=True)

    bin_index = min(bin_index, n_frames - 1)

    bin_index = int(bin_index)

    return bin_index

def frame_index_to_timestamp(frame_index, video_duration, n_frames=32):
    """
    Convert a frame index to a timestamp in seconds.
    1) Create bins by dividing the video duration into n_frames bins.
    2) Find the timestamp that corresponds to the bin.
    """
    video_duration = int(video_duration)
    if n_frames < 0:
        n_frames = video_duration

    bins = np.linspace(0, video_duration-1, n_frames)

    # e.g., video_duration = 200, n_frames: 32
    # bins
    # [  0.           6.41935484  12.83870968  19.25806452  25.67741935
    #   32.09677419  38.51612903  44.93548387  51.35483871  57.77419355
    #   64.19354839  70.61290323  77.03225806  83.4516129   89.87096774
    #   96.29032258 102.70967742 109.12903226 115.5483871  121.96774194
    #  128.38709677 134.80645161 141.22580645 147.64516129 154.06451613
    #  160.48387097 166.90322581 173.32258065 179.74193548 186.16129032
    #  192.58064516 199.        ]

    timestamp = bins[frame_index]

    timestamp = int(timestamp)

    return timestamp


class MomentDataset(Dataset):
    def __init__(self, args, data_path, video_dir=None, video_feature_dir=None, asr_dir=None, asr_feature_dir=None, n_model_frames=-1, task=None):
        prompt2video_anns = json.load(open(data_path, 'r'))

        self.args = args

        self.video_dir = video_dir
        self.video_feature_dir = video_feature_dir
        self.asr_dir = asr_dir
        self.asr_feature_dir = asr_feature_dir

        if video_dir is not None:
            self.video_dir = Path(video_dir)
            self.transform = clip.clip._transform(224)

        if video_feature_dir is not None:
            self.video_feature_dir = Path(video_feature_dir)

            assert self.video_feature_dir.exists(), f'video_feature_dir {self.video_feature_dir} does not exist'

        self.videoid2asr = {}
        if asr_dir is not None:
            self.asr_dir = Path(asr_dir)
            self.asr_feature_dir = Path(asr_feature_dir)

            assert self.asr_dir.exists(), self.asr_dir
            assert self.asr_feature_dir.exists(), self.asr_feature_dir

            self.videoid2asr = {}
            for asr_path in self.asr_dir.glob('*.srt'):
                video_id = asr_path.stem
                with open(asr_path, 'r') as f:
                    transcript_srt_str = f.read()

                all_subs = []
                for sub in srt.parse(transcript_srt_str):
                    all_subs.append(sub)

                self.videoid2asr[video_id] = all_subs

        self._all_prompts = list(prompt2video_anns.keys())

        self.n_model_frames = n_model_frames

        self.tasks = ['moment_retrieval', 'moment_segmentation', 'step_captioning']
        self.task = task

        import sys
        sys.path.append("./clip4caption/")
        from modules.tokenization import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        data = []

        n_prompts_with_relevant_videos = 0

        for i, (prompt, video_anns) in enumerate(prompt2video_anns.items()):

            has_relevant_videos = False
            for video_fname, video_ann in video_anns.items():
                if not video_ann['relevant']:
                    continue
                if not video_ann['clip']:
                    continue

                has_relevant_videos = True

                datum = {}

                datum['fname'] = video_fname

                datum['prompt'] = prompt

                video_duration = video_ann['v_duration']
                video_duration = round(video_duration)
                datum['video_duration'] = video_duration
                datum['n_model_frames'] = n_model_frames

                if self.n_model_frames > 0:
                    n_frames = self.n_model_frames
                else:
                    n_frames = video_duration

                if task == 'moment_retrieval':

                    task_datum = deepcopy(datum)
                    task_datum['task'] = task

                    original_bounds = []
                    approximate_bounds = []

                    moment_start = video_ann['bounds'][0]
                    moment_end = video_ann['bounds'][1]

                    start_frame = timestamp_to_frame_index(moment_start, video_duration=video_duration, n_frames=n_frames)
                    end_frame = timestamp_to_frame_index(moment_end, video_duration=video_duration, n_frames=n_frames)

                    task_datum['moment_retrieval_start_target'] = start_frame
                    task_datum['moment_retrieval_end_target'] = end_frame

                    original_bounds += [[moment_start, moment_end]]
                    approximate_bounds += [[frame_index_to_timestamp(start_frame, video_duration=video_duration, n_frames=n_frames),
                                            frame_index_to_timestamp(end_frame, video_duration=video_duration, n_frames=n_frames)]]

                    task_datum['original_bounds'] = original_bounds
                    task_datum['approximate_bounds'] = approximate_bounds

                    video_mask = torch.ones(n_frames, dtype=torch.long)
                    task_datum['video_mask'] = video_mask

                    moment_mask = torch.ones(n_frames, dtype=torch.long)
                    task_datum['moment_mask'] = moment_mask

                    data.append(task_datum)

                elif task == 'moment_segmentation':
                    if not args.end_to_end:
                        if len(video_ann['steps']) == 0:
                            continue

                    if 'train' in str(data_path):

                        moment_start = video_ann['bounds'][0]
                        moment_end = video_ann['bounds'][1]

                        moment_start_frame = timestamp_to_frame_index(moment_start, video_duration=video_duration, n_frames=n_frames)
                        moment_end_frame = timestamp_to_frame_index(moment_end, video_duration=video_duration, n_frames=n_frames)

                        all_boundaries = []
                        for step in video_ann['steps']:
                            all_boundaries += step['absolute_bounds']
                        all_boundaries = sorted(list(set(all_boundaries)))
                        all_boundaries_frames = [timestamp_to_frame_index(bound, video_duration=video_duration, n_frames=n_frames) for bound in all_boundaries]

                        if len(all_boundaries) <= 2:
                            continue

                        for i in range(len(all_boundaries) - 1):

                            task_datum = deepcopy(datum)
                            task_datum['task'] = task

                            start = all_boundaries[i]
                            end = all_boundaries[i + 1]
                            step_start_frame = timestamp_to_frame_index(start, video_duration=video_duration, n_frames=n_frames)
                            step_end_frame = timestamp_to_frame_index(end, video_duration=video_duration, n_frames=n_frames)

                            all_prev_boundaries = torch.zeros(n_frames, dtype=torch.long)
                            for b in all_boundaries[:i + 1]:
                                b_frame = timestamp_to_frame_index(b, video_duration=video_duration, n_frames=n_frames)
                                all_prev_boundaries[b_frame] = 1
                            task_datum['prev_boundary_mask'] = all_prev_boundaries

                            task_datum['moment_segmentation_target'] = step_end_frame

                            moment_mask = torch.zeros(n_frames, dtype=torch.long)
                            moment_mask[step_start_frame:moment_end_frame+1] = 1
                            task_datum['moment_mask'] = moment_mask

                            video_mask = torch.ones(n_frames, dtype=torch.long)
                            task_datum['video_mask'] = video_mask

                            task_datum['moment_bound_timestamps'] = [moment_start, moment_end]
                            task_datum['moment_bound_frames'] = [moment_start_frame, moment_end_frame]

                            task_datum['all_bound_frames'] = all_boundaries_frames

                            data.append(task_datum)

                    else:
                        all_boundaries = []
                        for step in video_ann['steps']:
                            all_boundaries += step['absolute_bounds']
                        all_boundaries = sorted(list(set(all_boundaries)))
                        all_boundaries_frames = [timestamp_to_frame_index(bound, video_duration=video_duration, n_frames=n_frames) for bound in all_boundaries]

                        task_datum = deepcopy(datum)
                        task_datum['task'] = task

                        moment_start = video_ann['bounds'][0]
                        moment_end = video_ann['bounds'][1]

                        moment_start_frame = timestamp_to_frame_index(moment_start, video_duration=video_duration, n_frames=n_frames)
                        moment_end_frame = timestamp_to_frame_index(moment_end, video_duration=video_duration, n_frames=n_frames)

                        task_datum['moment_bound_timestamps'] = [moment_start, moment_end]
                        task_datum['moment_bound_frames'] = [moment_start_frame, moment_end_frame]

                        moment_mask = torch.zeros(n_frames, dtype=torch.long)
                        moment_mask[moment_start_frame:moment_end_frame+1] = 1
                        task_datum['moment_mask'] = moment_mask

                        video_mask = torch.ones(n_frames, dtype=torch.long)
                        task_datum['video_mask'] = video_mask

                        task_datum['all_bound_frames'] = all_boundaries_frames

                        data.append(task_datum)

                elif task == 'step_captioning':
                    if not args.end_to_end:
                        if len(video_ann['steps']) == 0:
                            continue

                    target_text = []
                    original_bounds = []
                    approximate_bounds = []

                    moment_start = video_ann['steps'][0]['absolute_bounds'][0]
                    moment_end = video_ann['steps'][-1]['absolute_bounds'][1]

                    _moment_start = video_ann['bounds'][0]
                    _moment_end = video_ann['bounds'][1]

                    for step in video_ann['steps']:
                        step_start, step_end = step['absolute_bounds']
                        step_text = step['heading'].strip()

                        start_frame = timestamp_to_frame_index(step_start, video_duration=video_duration, n_frames=n_frames)
                        end_frame = timestamp_to_frame_index(step_end, video_duration=video_duration, n_frames=n_frames)

                        task_datum = deepcopy(datum)
                        task_datum['task'] = task

                        target_text = step_text

                        target_text = self.clip4cap_get_text(target_text)
                        task_datum["target_text_raw"] = step_text

                        task_datum["target_text"] = target_text

                        moment_mask = torch.zeros(n_frames, dtype=torch.long)
                        moment_mask[start_frame:end_frame] = 1
                        moment_mask[end_frame] = 1

                        task_datum['moment_mask'] = moment_mask

                        video_mask = torch.ones(n_frames, dtype=torch.long)
                        task_datum['video_mask'] = video_mask

                        data.append(task_datum)

            if has_relevant_videos:
                n_prompts_with_relevant_videos += 1

        self.data = data

        print(f'# {task} examples:', len([d for d in data if d['task'] == task]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]

        out = deepcopy(datum)

        if self.video_feature_dir is not None:
            video_fname = datum['fname']
            video_feature_path = self.video_feature_dir / f'{video_fname}.pt'
            video_features = torch.load(video_feature_path, map_location='cpu')

            if self.n_model_frames > 0:
                n_frames = video_features.shape[0]
                # Uniformly subsample via linspace
                if n_frames > self.n_model_frames:
                    frame_ids = np.linspace(
                        0, n_frames - 1, self.n_model_frames).astype(int)
                    frame_ids = torch.from_numpy(frame_ids)
                    video_features = video_features[frame_ids]
                else:
                    x = torch.zeros((self.n_model_frames, video_features.shape[1]))
                    count_embeds = [ 0 ] * self.n_model_frames
                    N: int = video_features.shape[0]

                    count_embeds = [ count_embeds[(j*self.n_model_frames) // N : ((j+1)*self.n_model_frames) // N] for j in range(N) ]

                    j = 0
                    for k in range(len(count_embeds)):
                        for _ in count_embeds[k]:
                            x[j] = video_features[k]
                            j += 1
                    
                    video_features = x.clone()

            out['vis_feats'] = video_features


        if len(self.videoid2asr) > 0:
            video_fname = datum['fname']
            video_id = video_fname.replace('.mp4', '')
            subs = self.videoid2asr[video_id]

            asr_feature_path = self.asr_feature_dir / f'{video_id}.pt'

            assert asr_feature_path.exists(), asr_feature_path

            asr_features = torch.load(asr_feature_path, map_location='cpu')

            # warping

            dim = asr_features.shape[1]
            len_vid = video_features.shape[0]

            warped_asr_embedding = torch.zeros(len_vid, dim).float()

            for i, sub in enumerate(subs):
                start, end = sub.start.seconds, sub.end.seconds

                warped_asr_embedding[start:end] = asr_features[i]

            if self.n_model_frames > 0:
                n_frames = warped_asr_embedding.shape[0]
                # Uniformly subsample via linspace
                if n_frames > self.n_model_frames:
                    frame_ids = np.linspace(
                        0, n_frames - 1, self.n_model_frames).astype(int)
                    frame_ids = torch.from_numpy(frame_ids)
                    warped_asr_embedding = warped_asr_embedding[frame_ids]
                else:
                    x = torch.zeros((self.n_model_frames, warped_asr_embedding.shape[1]))
                    count_embeds = [ 0 ] * self.n_model_frames
                    N: int = warped_asr_embedding.shape[0]

                    count_embeds = [ count_embeds[(j*self.n_model_frames) // N : ((j+1)*self.n_model_frames) // N] for j in range(N) ]

                    j = 0
                    for k in range(len(count_embeds)):
                        for _ in count_embeds[k]:
                            x[j] = warped_asr_embedding[k]
                            j += 1
                    
                    warped_asr_embedding = x.clone()

            out['asr_feats'] = warped_asr_embedding

        return out

    def collate_fn(self, batch):
        out_batch = {}

        if 'target_text' in batch[0]:
            target_text = [datum['target_text'] for datum in batch]
            out_batch['target_text'] = target_text

        if 'target_text_raw' in batch[0]:
            target_text_raw = [datum['target_text_raw'] for datum in batch]
            out_batch['target_text_raw'] = target_text_raw


        if 'vis_feats' in batch[0]:

            if self.n_model_frames > 0:
                video_features = []
                for datum in batch:
                    video_features += [datum['vis_feats']]
                video_features = torch.stack(video_features)
                video_features = video_features.float()
                assert video_features.shape[0] == len(batch)
            else:
                video_feat_lens = []
                for datum in batch:
                    video_feat_lens.append(datum['vis_feats'].shape[0])
                max_video_feat_len = max(video_feat_lens)
                video_features = []
                for datum in batch:
                    n_frames, dim = datum['vis_feats'].shape
                    n_pad = max_video_feat_len - n_frames
                    video_features.append(torch.cat([datum['vis_feats'], torch.zeros(n_pad, dim)], dim=0))
                video_features = torch.stack(video_features)

            out_batch['vis_feats'] = video_features


            if self.n_model_frames > 0:
                video_mask = [datum['video_mask'] for datum in batch]
                video_mask = torch.stack(video_mask)
            else:
                video_mask = []
                for datum in batch:
                    n_pad = max_video_feat_len - datum['vis_feats'].shape[0]
                    video_mask.append(torch.cat([datum['video_mask'], torch.zeros(n_pad)], dim=0))
                video_mask = torch.stack(video_mask)

            out_batch['vis_mask'] = video_mask.long()


            if self.n_model_frames > 0:
                moment_mask = [datum['moment_mask'] for datum in batch]
                moment_mask = torch.stack(moment_mask)

            else:
                moment_mask = []
                for datum in batch:
                    n_pad = max_video_feat_len - datum['vis_feats'].shape[0]
                    moment_mask.append(torch.cat([datum['moment_mask'], torch.zeros(n_pad)], dim=0))
                moment_mask = torch.stack(moment_mask)

            out_batch['moment_mask'] = moment_mask.long()


            if 'moment_retrieval_start_target' in batch[0]:
                moment_retrieval_start_target = [datum['moment_retrieval_start_target'] for datum in batch]
                out_batch['moment_retrieval_start_target'] = torch.LongTensor(moment_retrieval_start_target)

            if 'moment_retrieval_end_target' in batch[0]:
                moment_retrieval_end_target = [datum['moment_retrieval_end_target'] for datum in batch]
                out_batch['moment_retrieval_end_target'] = torch.LongTensor(moment_retrieval_end_target)

        if 'prev_boundary_mask' in batch[0]:
            
            if self.n_model_frames > 0:
                prev_boundary_mask = [datum['prev_boundary_mask'] for datum in batch]
                prev_boundary_mask = torch.stack(prev_boundary_mask).long()
                out_batch['prev_boundary_mask'] = prev_boundary_mask
            else:
                prev_boundary_mask = []
                for datum in batch:
                    n_pad = max_video_feat_len - datum['vis_feats'].shape[0]
                    prev_boundary_mask.append(torch.cat([datum['prev_boundary_mask'], torch.zeros(n_pad)], dim=0))
                out_batch['prev_boundary_mask'] = torch.stack(prev_boundary_mask).long()

        if 'moment_segmentation_target' in batch[0]:
            moment_segmentation_target = [datum['moment_segmentation_target'] for datum in batch]
            out_batch['moment_segmentation_target'] = torch.LongTensor(moment_segmentation_target)

        if 'asr_feats' in batch[0]:
            if self.n_model_frames > 0:
                asr_features = [datum['asr_feats'] for datum in batch]
                asr_features = torch.stack(asr_features).float()
                out_batch['asr_feats'] = asr_features
            else:
                asr_features = []
                for datum in batch:
                    n_pad = max_video_feat_len - datum['vis_feats'].shape[0]
                    dim = datum['asr_feats'].shape[1]
                    asr_features.append(torch.cat([datum['asr_feats'], torch.zeros(n_pad, dim)], dim=0))
                out_batch['asr_feats'] = torch.stack(asr_features).float()

        if 'moment_bound_timestamps' in batch[0]:
            moment_bound_timestamps = [datum['moment_bound_timestamps'] for datum in batch]
            out_batch['moment_bound_timestamps'] = torch.LongTensor(moment_bound_timestamps)

        if 'moment_bound_frames' in batch[0]:
            moment_bound_frames = [datum['moment_bound_frames'] for datum in batch]
            out_batch['moment_bound_frames'] = torch.LongTensor(moment_bound_frames)

        if 'all_bound_frames' in batch[0]:
            all_bound_frames = [datum['all_bound_frames'] for datum in batch]
            out_batch['all_bound_frames'] = all_bound_frames

        out_batch['video_duration'] = [datum['video_duration'] for datum in batch]

        out_batch['video_fnames'] = [datum['fname'] for datum in batch]
        out_batch['tasks'] = [datum['task'] for datum in batch]
        out_batch['prompts'] = [datum['prompt'] for datum in batch]

        clip_text_ids = clip.tokenize(out_batch['prompts'])
        out_batch['clip_text_ids'] = clip_text_ids

        return out_batch

    def clip4cap_get_text(self, caption=None):
        k = 1
        pairs_text = np.zeros((k, self.args.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.args.max_words), dtype=np.long)
        pairs_output_caption_ids = np.zeros((k, self.args.max_words), dtype=np.long)
        pairs_decoder_mask = np.zeros((k, self.args.max_words), dtype=np.long)

        words = []
        words = ["[CLS]"] + words
        total_length_with_CLS = self.args.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + ["[SEP]"]


        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        while len(input_ids) < self.args.max_words:
            input_ids.append(0)
        assert len(input_ids) == self.args.max_words

        pairs_text[0] = np.array(input_ids)

        if caption is not None:
            caption_words = self.tokenizer.tokenize(caption)

        if len(caption_words) > total_length_with_CLS:
            caption_words = caption_words[:total_length_with_CLS]
        input_caption_words = ["[CLS]"] + caption_words
        output_caption_words = caption_words + ["[SEP]"]

        input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
        output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
        decoder_mask = [1] * len(input_caption_ids)
        while len(input_caption_ids) < self.args.max_words:
            input_caption_ids.append(0)
            output_caption_ids.append(0)
            decoder_mask.append(0)
        assert len(input_caption_ids) == self.args.max_words
        assert len(output_caption_ids) == self.args.max_words
        assert len(decoder_mask) == self.args.max_words

        pairs_input_caption_ids[0] = np.array(input_caption_ids)
        pairs_output_caption_ids[0] = np.array(output_caption_ids)
        pairs_decoder_mask[0] = np.array(decoder_mask)

        return pairs_text, np.array([]), np.array([]), np.array([]), np.array([]), \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, []

def get_moment_loader(args, split='train', batch_size=32, task='moment_retrieval'):

    assert task in ['moment_retrieval', 'moment_segmentation', 'step_captioning'], task

    if 'temp' in str(args.data_dir):
        data_path = Path(args.data_dir) / f'temp_data_{split}.json'
    else:
        data_path = Path(args.data_dir) / f'all_data_{split}.json'

    dataset = MomentDataset(
        args,
        data_path=data_path,
        video_dir=None,
        video_feature_dir=args.video_feature_dir,
        asr_dir=args.asr_dir,
        asr_feature_dir=args.asr_feature_dir,
        n_model_frames=args.n_model_frames,
        task=task,
    )

    shuffle = True if split == 'train' else False

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=shuffle)
    else:
        sampler = None

    if split == 'train':
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            num_workers=args.num_workers,
            drop_last=False
        )

    loader.task = task

    return loader

class MultitaskLoader(object):
    def __init__(self, loaders, shuffle=True, drop_last=False, sampling='roundrobin', n_batches=None, verbose=True):
        self.loaders = loaders
        self.verbose = verbose
        self.task2len = {loader.task: len(loader) for loader in self.loaders}
        if self.verbose:
            print('Task2len:', self.task2len)
        self.task2loader = {loader.task: loader for loader in self.loaders}

        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sampling = sampling
        self.epoch_tasks = None
        self.n_batches = n_batches
        self.set_epoch(0)

    def __iter__(self):
        self.task2iter = {loader.task: iter(loader) for loader in self.loaders}

        return self

    def set_epoch(self, epoch):
        for loader in self.loaders:
            if hasattr(loader, 'set_epoch'):
                loader.set_epoch(epoch)

        if self.sampling == 'roundrobin':
            epoch_tasks = []
            for task, loader in self.task2loader.items():
                n_batches = len(loader)
                epoch_tasks.extend([task]*n_batches)
        elif self.sampling == 'balanced':
            if self.n_batches is None:
                n_batches = sum(self.task2len.values()) // len(self.loaders)
            else:
                n_batches = self.n_batches
            if self.verbose:
                print('# batches:', n_batches)
            epoch_tasks = []
            for task, loader in self.task2loader.items():
                epoch_tasks.extend([task]*n_batches)

        if self.shuffle:
            random.Random(epoch).shuffle(epoch_tasks)
        self.epoch_tasks = epoch_tasks
        if self.verbose:
            print('# epoch_tasks:', len(self.epoch_tasks))

    def __next__(self):
        if len(self.epoch_tasks) > 0:
            task = self.epoch_tasks.pop()
            loader_iter = self.task2iter[task]
            return next(loader_iter)
        else:
            raise StopIteration

    def __len__(self):
        return len(self.epoch_tasks)
