from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle5 as pickle
import pandas as pd
from collections import defaultdict
import json
import random
from tqdm import tqdm
from scipy import sparse
import glob
import torch

class HODINI_Feats_DataLoader(Dataset):
    """Implementation of the dataloader for MSRVTT. Mainly used in the model training and evaluation.
    Params:
        json_path: Path to the MSRVTT_data.json file.
        features_path: Path to the extracted feature file.
        tokenizer: Tokenizer used for tokenizing the caption.
        max_words: Max word length retained. Any more than the value will be truncated. Default: 30
        feature_framerate: sampling rate in second. Default: 1.0
        max_frames: Max frame sampled. Any more than the value will be ignored. Default: 100
        split_type: Either "train", "val", or "test". Default: ""
    """
    def __init__(
            self,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            split_type="",
    ):
        assert split_type in ["train", "val", "test"]
        data = json.load(open(f"{json_path}/all_data_{split_type}.json", 'r'))
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer

        self.feature_size = 512


        self.sentences_dict = [  ]
        self.video_data = { }

        for prompt in data:
            for video in data[prompt]:
                steps = data[prompt][video]["steps"]

                if len(steps) > 0:
                    # target_steps = []
                    # target_frames = []

                    for i, step in enumerate(steps):
                        # target_steps.append(step["heading"])
                        # target_frames.append(round(np.average(step["absolute_bounds"])))

                        self.sentences_dict.append((f"{video}_{i}", step["heading"]))
                        self.video_data[f"{video}_{i}"] = (f"{features_path}{video}.pt", step["absolute_bounds"])

            # self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0])

        self.sample_len = len(self.sentences_dict)

    def __len__(self):
        return self.sample_len

    def _get_text(self, caption=None):
        k = 1
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)

        words = []
        words = ["[CLS]"] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + ["[SEP]"]


        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
        assert len(input_ids) == self.max_words

        pairs_text[0] = np.array(input_ids)

        # For generate captions
        if caption is not None:
            caption_words = self.tokenizer.tokenize(caption)
        # else:
        #     caption_words = self._get_single_text(video_id)
        if len(caption_words) > total_length_with_CLS:
            caption_words = caption_words[:total_length_with_CLS]
        input_caption_words = ["[CLS]"] + caption_words
        output_caption_words = caption_words + ["[SEP]"]

        # For generate captions
        input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
        output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
        decoder_mask = [1] * len(input_caption_ids)
        while len(input_caption_ids) < self.max_words:
            input_caption_ids.append(0)
            output_caption_ids.append(0)
            decoder_mask.append(0)
        assert len(input_caption_ids) == self.max_words
        assert len(output_caption_ids) == self.max_words
        assert len(decoder_mask) == self.max_words

        pairs_input_caption_ids[0] = np.array(input_caption_ids)
        pairs_output_caption_ids[0] = np.array(output_caption_ids)
        pairs_decoder_mask[0] = np.array(decoder_mask)

        return pairs_text, np.array([]), np.array([]), np.array([]), np.array([]), \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, []

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_video(self, video_id):
        video_mask = np.zeros((1, self.max_frames), dtype=np.long)
        
        max_video_length = [0] * 1

        video = np.zeros((1, self.max_frames, self.feature_size), dtype=np.float)

        # video_slice = self.feature_dict[video_id]
        p, bounds = self.video_data[video_id]
        video_slice = torch.load(p)
        # print(video_slice.shape, bounds)

        # try:
        video_slice = video_slice[int(bounds[0]):int(bounds[1])]
        # except:
        #     print("---------------------------------------------------------------------------------------------")
        #     print(video_slice.shape, bounds, type(bounds[0]), type(bounds[1]))

        if self.max_frames < video_slice.shape[0]:
            video_slice = video_slice[:self.max_frames]
            # idx = np.round(np.linspace(0, video_slice.shape[0]-1, self.max_frames)).astype(int)
            # video_slice = video_slice[idx]
        else:
            x = torch.zeros((self.max_frames, self.feature_size))
            count_embeds = [ 0 ] * self.max_frames
            N: int = video_slice.shape[0]

            count_embeds = [ count_embeds[(i*self.max_frames) // N : ((i+1)*self.max_frames) // N] for i in range(N) ]

            j = 0
            for i in range(len(count_embeds)):
                for _ in count_embeds[i]:
                    x[j] = video_slice[i]
                    j += 1
            
            video_slice = x.clone()
            

        slice_shape = video_slice.shape
        max_video_length[0] = max_video_length[0] if max_video_length[0] > slice_shape[0] else slice_shape[0]
        if len(video_slice) < 1:
            print("video_id: {}".format(video_id))
        else:
            video[0][:slice_shape[0]] = video_slice

        return video, video_mask, np.array([]), np.array([])

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]
        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, \
        pairs_input_caption_ids, pairs_decoder_mask, \
        pairs_output_caption_ids, choice_video_ids = self._get_text(caption)

        video, video_mask, masked_video, video_labels_index = self._get_video(video_id)
        

        pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, masked_video, video_labels_index = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])

        
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids