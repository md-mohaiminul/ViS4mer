import torch
import os
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
set_seed(1112)

DATA_ROOT = '/playpen-storage/mmiemon/lvu/data'
duration_data = pd.read_csv('/playpen-storage/mmiemon/lvu/data/CMD/metadata/durations.csv').set_index('videoid')

class CustomDataset(Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split

        self.videos = []
        self.labels = []
        self.starts = []

        csv_file = f'{DATA_ROOT}/lvu_1.0/{args.long_term_task}/{split}.csv'
        with open(csv_file, 'r') as f:
            f.readline()
            for line in f:
                video_id = line.split()[-2].strip()
                if not os.path.exists(f'{DATA_ROOT}/vit_features_spatial/{video_id}.npy'):
                    print("Features not found for video : ", video_id)
                    continue

                if args.long_term_task == 'view_count':
                    label = float(np.log(float(line.split()[0])))

                    # make zero-mean
                    label -= 11.76425435683139
                elif args.long_term_task == 'like_ratio':
                    items = line.split()
                    like, dislike = float(items[0]), float(items[1])
                    label = like / (like + dislike) * 10.0

                    # make zero-mean
                    label -= 9.138220535629456
                else:
                    label = int(line.split()[0])

                duration = duration_data.loc[video_id]['duration']

                self.videos.append(video_id)
                self.starts.append(0)
                self.labels.append(label)

                for start in range(1, duration-args.l_secs+1):
                    self.videos.append(video_id)
                    self.starts.append(start)
                    self.labels.append(label)

            print('Total videos in ', split, len(set(self.videos)))
            print('Total spans ', split, len(self.videos))

    def __len__(self):
        if self.split == 'train':
            return len(set(self.videos))
        else:
            return len(self.videos)

    def __getitem__(self, idx):
        if self.split == 'train':
            idx = random.randint(0, len(self.videos)-1)

        if self.args.feature_type == 'vit_spatial':
            video_features = np.load(f'{DATA_ROOT}/vit_features_spatial/{self.videos[idx]}.npy')
            x = np.zeros((self.args.l_secs, 197, 1024))
            for i in range(self.starts[idx], min(self.starts[idx] + self.args.l_secs, video_features.shape[0])):
                x[i - self.starts[idx]] = video_features[i]
            x = np.reshape(x,(x.shape[0]* x.shape[1], 1024))

        #you can add support for other types of features smilarly

        return self.videos[idx], x, self.labels[idx]