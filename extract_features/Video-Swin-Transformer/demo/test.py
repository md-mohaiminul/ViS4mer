from mmaction.apis import init_recognizer, inference_recognizer
from mmaction.models import build_model

import skvideo.io
from moviepy.editor import *
import cv2
import numpy as np
import torch

#video = '/playpen-storage/mmiemon/lvu/data/breakfast/BreakfastII_15fps_qvga_sync/P03/cam01/P03_friedegg.avi'
video = 'demo.mp4'
clip = VideoFileClip(video)
n_frames = clip.duration * clip.fps
print(n_frames)

frames = []
for i in range (32):
    image = cv2.resize(clip.get_frame(i/clip.fps), (224, 224), interpolation=cv2.INTER_AREA)
    frames.append(image)
frames = np.asarray(frames) / 255.0
print(frames.shape)
frames = torch.from_numpy(frames.transpose([3, 0, 1, 2])).float()
frames = torch.unsqueeze(frames, 0)
print(frames.shape)

config_file = '../configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py'
# # download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../checkpoints/swin_base_patch244_window877_kinetics600_22k.pth'

device = 'cuda:0' # or 'cpu'
model = init_recognizer(config_file, checkpoint_file, device=device)

features = model.extract_feat(frames.to(device))
print(features.shape)
print(torch.max(features), torch.min(features))

label = 'label_map_k600.txt'
results, features = inference_recognizer(model, video, label, outputs='backbone')

# show the results
for result in results:
    print(f'{result[0]}: ', result[1])

print(features['backbone'].shape)