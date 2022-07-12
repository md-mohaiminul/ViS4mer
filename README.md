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



