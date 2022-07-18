import torch
import os
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
set_seed(1112)

DATA_ROOT = '/playpen-storage/mmiemon/lvu/data'

class CustomDataset(Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split
        self.videos = []
        self.labels = []
        csv_file = f'data/Breakfast/{split}.csv'
        with open(csv_file, 'r') as f:
            f.readline()
            for line in f:
                video_id = line.split(',')[0]
                if video_id in ['P28-cam01-P28_cereals', 'P27-stereo-P27_milk_ch0', 'P28-cam02-P28_cereals']:
                    continue
                label = int(line.split(',')[-1])
                self.videos.append(video_id)
                self.labels.append(label)
            print('Total videos in ', split, len(self.videos))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_features = np.load(f'{DATA_ROOT}/breakfast/video_swin_features/{self.args.feature_type}/{self.videos[idx]}.npy')

        if video_features.shape[0] < self.args.l_secs:
            step = video_features.shape[0] / float(self.args.l_secs)
            indices = np.arange(0, video_features.shape[0], step, dtype=np.float32).astype(np.int32)
            video_features = video_features[indices]

        elif video_features.shape[0] > self.args.l_secs:
            if self.split == 'train':
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
# args.feature_type = 'temporal_mean_pooling'
# args.l_secs = 64
# trainset = CustomDataset(args=args, split='train')
# print(len(trainset))
# a,x,b = next(iter(trainset))
# print(x.shape)
