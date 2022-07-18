import torch
import os
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import argparse
import json

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
set_seed(1112)

DATA_ROOT = '/playpen-storage/mmiemon/lvu/data'

f = open('data/COIN/COIN.json')
data = json.load(f)['database']

class CustomDataset(Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split
        self.videos = []
        self.labels = []

        for video_id in data:
            if video_id in ['AfiVmAjfTNs', 'cjwtcDKTQM8']:
                continue
            if data[video_id]['subset'] == split:
                self.videos.append(video_id)
                self.labels.append(data[video_id]['recipe_type'])

        print('Total videos in ', split, len(self.videos))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_features = np.load(f'{DATA_ROOT}/COIN/features/{self.args.feature_type}/{self.videos[idx]}.npy')

        if video_features.shape[0] < self.args.l_secs:
            step = video_features.shape[0] / float(self.args.l_secs)
            indices = np.arange(0, video_features.shape[0], step, dtype=np.float32).astype(np.int32)
            video_features = video_features[indices]

        elif video_features.shape[0] > self.args.l_secs:
            if self.split == 'training':
                indices = random.sample(range(0, video_features.shape[0]), self.args.l_secs)
                indices.sort()
                video_features = video_features[indices]
            else:
                step = video_features.shape[0] / float(self.args.l_secs)
                indices = np.arange(0, video_features.shape[0], step, dtype=np.float32).astype(np.int32)
                video_features = video_features[indices]

        video_features = np.reshape(video_features,(video_features.shape[0]* video_features.shape[1], 1024))

        return self.videos[idx], video_features, self.labels[idx]

# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# args = parser.parse_args()
# args.feature_type = 'spatial_mean_pooling'
# args.l_secs = 64
# trainset = CustomDataset(args=args, split='testing')
# print(len(trainset))
# v, f, l = next(iter(trainset))
# print(f.shape, l)
