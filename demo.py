from models import ViS4mer
import torch

model = ViS4mer(
            d_input=1024,
            l_max=2048,
            d_output=10,
            d_model=1024,
            n_layers=3,
            dropout=0.1,
            prenorm=True,
        )
model.cuda()
inputs = torch.randn(32, 2048, 1024).cuda()
outputs = model(inputs)  #[32, 10]

print(outputs.shape)