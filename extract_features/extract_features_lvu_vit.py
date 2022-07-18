import timm
import numpy as np
import pandas as pd
import skvideo.io
import cv2
import torch
import os
import glob
import torch.nn as nn
import random
import pickle

DATA_ROOT = '/playpen-storage/mmiemon/lvu/'

duration_data = pd.read_csv('../data/lvu_1.0/lvu_durations.csv').set_index('videoid')

def get_video(video_path):
    video = skvideo.io.vread(video_path)
    frames = []
    for i in range(video.shape[0]):
        image = cv2.resize(video[i], (224, 224), interpolation=cv2.INTER_AREA)
        frames.append(image)
    frames = np.asarray(frames) / 255.0
    return frames

model = timm.create_model('vit_large_patch16_224_in21k', num_classes = 0, pretrained=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

#check if everything is working fine
x = torch.randn(2, 3, 224, 224).to(device)
y = model.forward_features(x)
print(y.shape)   #this should be of dimention [2,197,1024]

all_ids = []
for csv in glob.glob(f'{DATA_ROOT}data/lvu_1.0/*/*.csv'):
    with open(csv, 'r') as f:
        f.readline()
        for line in f:
            video_id = line.split()[-2].strip()
            all_ids.append(video_id)

print('Total videos: ', len(all_ids))
print('Total unique videos: ', len(set(all_ids)))

random.shuffle(all_ids) #helps if you want to run multiple instances parallelly

cnt = 0
for video_id in all_ids:
    dest = f'{DATA_ROOT}data/covnext_feature/{video_id}.npy' #destination to save features
    if os.path.exists(dest):
        video_fp = f'{DATA_ROOT}data/mc_videos/{video_id}.mp4' #destination of source video
        if not os.path.exists(video_fp):
            print('Video not found :', video_id)
        if os.path.exists(video_fp):
            video = get_video(video_fp)
            video = torch.from_numpy(video.transpose([0, 3, 1, 2])).float()
            duration = duration_data.loc[video_id]['duration']
            print(cnt, video_id, video.shape, duration)

            features = np.zeros((duration+1, 197, 1024))

            for i in range(int(duration)):
                idx = int(video.shape[0] / duration * i)
                x = torch.unsqueeze(video[idx], 0).to(device)
                x = model.forward_features(x)
                features[i] = x.detach().cpu().numpy()

            x = model.forward_features(torch.unsqueeze(video[-1], 0).to(device))
            features[duration] = x.detach().cpu().numpy()

            np.save(dest, features)
            cnt += 1
