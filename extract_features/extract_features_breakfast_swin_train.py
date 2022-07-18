import torch
import os
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from moviepy.editor import *
import cv2
import numpy as np
from mmaction.apis import init_recognizer, inference_recognizer
from mmaction.models import build_model
from einops import rearrange, reduce, repeat

config_file = 'Video-Swin-Transformer/configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py'
# # download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'Video-Swin-Transformer/checkpoints/swin_base_patch244_window877_kinetics600_22k.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = init_recognizer(config_file, checkpoint_file, device=device)

DATA_ROOT = '/playpen-storage/mmiemon/lvu/data'

all_ids = []
csv_file = f'../data/Breakfast/train.csv'
with open(csv_file, 'r') as f:
    f.readline()
    for line in f:
        video_id = line.split(',')[0]
        all_ids.append(video_id)

print('total files', len(set(all_ids)))
random.shuffle(all_ids)

for cnt, video_id in enumerate(all_ids):
    dest_mean = f'{DATA_ROOT}/breakfast/video_swin_features/temporal_mean_pooling/{video_id}.npy'

    if not os.path.exists(dest_mean):
        if video_id in ['P28-cam01-P28_cereals', 'P27-stereo-P27_milk_ch0', 'P28-cam02-P28_cereals']:
            continue
        file = video_id.split('.')[0].replace('-', '/')
        file = f'{DATA_ROOT}/breakfast/BreakfastII_15fps_qvga_sync/{file}.avi'
        clip = VideoFileClip(file)
        n_frames = int(clip.duration * clip.fps)
        n_segments = 512
        segment_length = 32

        if n_frames < (n_segments+segment_length):
            starts = [i for i in range(n_frames-segment_length)]
        else:
            step = (n_frames - segment_length) / float(n_segments)
            starts = np.arange(0, n_frames - segment_length, step=step)

        mean_features = []
        for start in starts:
            start = int(start)
            print(cnt, file, start, '/', n_frames)
            frames = []
            for i in range(start, start + segment_length):
                image = cv2.resize(clip.get_frame(i / clip.fps), (224, 224), interpolation=cv2.INTER_AREA)
                frames.append(image)
            frames = np.asarray(frames) / 255.0
            frames = torch.from_numpy(frames.transpose([3, 0, 1, 2])).float()
            frames = torch.unsqueeze(frames, 0)
            features = torch.squeeze(model.extract_feat(frames.to(device))).detach().cpu().numpy()

            mean = reduce(features, 'c t h w -> c h w', 'mean')
            mean = rearrange(mean, 'c h w-> (h w) c')
            mean_features.append(mean)

        mean_features = np.asarray(mean_features)

        print(cnt, file)
        print(mean_features.shape)

        np.save(dest_mean, mean_features)