import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class nBRC(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super(nBRC, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.Ua = nn.Linear(self.input_dim, self.hidden_dim)
        self.Wa = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.Uc = nn.Linear(self.input_dim, self.hidden_dim)
        self.Wc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.U = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, x, h):
        a = 1 + torch.tanh(self.Ua(x) + self.Wa(h))
        c = torch.sigmoid(self.Uc(x) + self.Wc(h))
        return c * h + (1 - c) * torch.tanh(self.U(x) + a * h)


class nBRCModel(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: list, full: bool = True):
        super(nBRCModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.full = full

        layers = []
        for i, hidden_dim in enumerate(self.hidden_dims):
            layers.append(
                nBRC(input_dim=self.input_dim if i == 0 else self.hidden_dims[i - 1],
                     hidden_dim=hidden_dim)
            )
        self.nBRCLayers = nn.ModuleList(layers)
        self.fc = nn.Linear(self.hidden_dims[-1], 5)

    def forward(self, x):
        B, seq_len, _ = x.size()

        for layer in self.nBRCLayers:
            h = torch.zeros((B, self.hidden_dims[0]))
            layer_output = []

            for t in range(seq_len):
                h = layer(x[:, t, :], h)
                layer_output.append(h.unsqueeze(1))
            x = torch.cat(layer_output, dim=1)

        return self.fc(x) if self.full else self.fc(x[:, -1, :])


def generate_data(B, N, T):
    x1 = -1 * torch.ones((B * T))
    idx = [t for tB
           in [np.sort(np.random.choice(range(T - N), size=5, replace=False)) + (i * T) for i in range(B)]
           for t in tB]
    x1[idx] = 0
    x1[-1] = 1
    x2 = torch.rand(B * T)
    yT = x2[idx].view(B, 5)
    x1 = x1.view(B, T)
    x2 = x2.view(B, T)
    x = torch.stack([x1, x2], dim=2)
    return x, yT


if __name__ == '__main__':

    # Denoising setup
    B = 100
    N = 200
    T = 400
    total_steps = 30_000
    device = 'cpu'

    # Init model
    model = nBRCModel(input_dim=2, hidden_dims=[100, 100, 100, 100], full=False)
    model.to(device)
    optim = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss(reduction='mean')

    with tqdm(total=total_steps, desc='Train: ') as pbar:

        for step in range(total_steps):
            # Generate data
            x, yT = generate_data(B, N, T)
            x = x.to(device)
            yT = yT.to(device)

            # Make prediction
            y = model(x)

            # Calculate loss
            loss = criterion(y, yT)

            # Perform weight update
            loss.backward()
            optim.step()
            optim.zero_grad()

            # Update progress bar
            pbar.set_postfix(loss=loss.item())
            pbar.update()
