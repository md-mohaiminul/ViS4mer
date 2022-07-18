# ViS4mer

This is an official pytorch implementation of our ECCV 2022 paper [Long Movie Clip Classification with State-Space Video Models](https://arxiv.org/abs/2204.01692). In this repository, we provide PyTorch code for training and testing our proposed ViS4mer model. ViS4mer is an efficient video recognition model that achieves state-of-the-art results on several long-range video understanding bechmarks such as [LVU](https://arxiv.org/abs/2106.11310), [Breakfast](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/), and [Coin](https://coin-dataset.github.io).

If you find ViS4mer useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@article{islam2022long,
  title={Long movie clip classification with state-space video models},
  author={Islam, Md Mohaiminul and Bertasius, Gedas},
  journal={arXiv preprint arXiv:2204.01692},
  year={2022}
}
```

# Installation

This repository requires Python 3.8+ and Pytorch 1.9+. 

- Create a conda virtual environment and activate it.
```
conda create --name py38 python=3.8
conda activate py38
```
- Install the package listed in `requirements.txt`
- The S4 layer requires "Cauchy Kernel" and we used the CUDA version. This can be installed by following commands.
```
cd extensions/cauchy
python setup.py install
```
- Install [Pykeops](https://www.kernel-operations.io/keops/index.html) by running `pip install pykeops==1.5 cmake`

For more details of installation regarding S4 layer, please follow [this](https://github.com/HazyResearch/state-spaces).

# Demo
You can use the model as follows:

```python
import torch
from models import ViS4mer

model = ViS4mer(d_input=1024, l_max=2048, d_output=10, d_model=1024, n_layers=3)
model.cuda()

inputs = torch.randn(32, 2048, 1024).cuda() #[batch_size, seq_len, input_dim]
outputs = model(inputs)  #[32, 10]
```

# Run on [LVU](https://arxiv.org/abs/2106.11310) dataset

- Dataset splits are provided `data/lvu_1.0`. Otherwise, you can also download [here](https://github.com/chaoyuaw/lvu).
- You can download videos from youtube using [`youtube-dl`](https://pypi.org/project/youtube_dl/). `download_videos.py` provides code for downloading videos using `youtube_dl`. Alternatively, you can acquire the videos from [here](https://www.robots.ox.ac.uk/~vgg/research/condensed-movies/).
- We used `ImageNet21k` pretrained ViT dense features from [`timm`](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py). Particularly, we used `vit_large_patch16_224_in21k` ViT model. Following provides code for extracting features for LVU dataset.
 ```extract_features/extract_features_lvu_vit.py```
- Finally, you can run the ViS4mer model on LVU tasks using `run_lvu.py`. Particularly, we used 4 GPUs and the following command.
 ```CUDA_VISIBLE_DEVICES=0,1,2,3 python run_lvu.py```
 
 # Run on [Breakfast](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/) dataset

- Download the [Breakfast]((https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/) dataset.
- We used [`VideoSwin`](https://github.com/SwinTransformer/Video-Swin-Transformer) features for the Breakfast dataset. Particularly, we used `swin_base_patch244_window877_kinetics600_22k` prtrained model. Following files provide code for extracting features for the Breakfast dataset train and test split respectively.
```
extract_features/extract_features_breakfast_swin_train.py
extract_features/extract_features_breakfast_swin_test.py
```
- Finally, you can run the ViS4mer model on Breakfast dataset using `run_breakfast.py`. Particularly, we used 4 GPUs and the following command.
 ```CUDA_VISIBLE_DEVICES=0,1,2,3 python run_breakfast.py```


