import torch
from models import ViS4mer

model = ViS4mer(d_input=1024, l_max=2048, d_output=10, d_model=1024, n_layers=3)
model.cuda()

inputs = torch.randn(32, 2048, 1024).cuda() #[batch_size, seq_len, input_dim]
outputs = model(inputs)  #[32, 10]

print(outputs.shape)