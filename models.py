import torch
import torch.nn as nn
from s4 import S4

class ViS4mer(nn.Module):

    def __init__(
            self,
            d_input,
            l_max,
            d_output,
            d_model,
            n_layers,
            dropout,
            prenorm,
    ):
        super().__init__()

        self.prenorm = prenorm
        self.d_model = d_model
        self.d_input = d_input

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.gelus = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4(H=d_model, l_max=l_max, dropout=dropout, transposed=True)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))
            self.pools.append(nn.AvgPool1d(2))
            self.linears.append(nn.Linear(d_model, int(d_model/2)))
            self.gelus.append(nn.GELU())
            d_model = int(d_model/2)
            l_max = int(l_max/2)

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = x.to(torch.float32)
        if self.d_model != self.d_input:
            x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout, pool,linear, gelu in \
                zip(self.s4_layers, self.norms, self.dropouts, self.pools, self.linears, self.gelus):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)
            # Dropout on the output of the S4 block
            z = dropout(z)
            # Residual connection
            x = z + x
            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

            #pooling layer
            x = pool(x)

            # MLP
            x = x.transpose(-1, -2)
            x = linear(x)
            x = gelu(x)
            x = x.transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)
        #x = x.max(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x