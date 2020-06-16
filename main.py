import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class nBRC(nn.Module):

    def __init__(self, input_dims: int, hidden_dims: int):
        super(nBRC, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims

        self.Ua = nn.Linear(self.input_dims, self.hidden_dims)
        self.Wa = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.Uc = nn.Linear(self.input_dims, self.hidden_dims)
        self.Wc = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.U = nn.Linear(self.input_dims, self.hidden_dims)

    def forward(self, x, h):
        a = 1 + torch.tanh(self.Ua(x) + self.Wa(h))
        c = torch.sigmoid(self.Uc(x) + self.Wc(h))
        return c * h + (1 - c) * torch.tanh(self.U(x) + a * h)


class nBRCModel(nn.Module):

    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int, num_layers: int, full: bool = True):
        super(nBRCModel, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.full = full

        self.nBRCLayers = nn.ModuleList(
            [nBRC(input_dims=self.input_dims if i == 0 else self.hidden_dims,
                  hidden_dims=self.hidden_dims)
             for i in range(num_layers)]
        )
        self.fc = nn.Linear(self.hidden_dims, self.output_dims)

    def forward(self, x):
        B, seq_len, _ = x.size()

        for layer in self.nBRCLayers:
            h = torch.zeros((B, self.hidden_dims), device=x.device)
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
    x2 = torch.rand(B * T)
    yT = x2[idx].view(B, 5)
    x1 = x1.view(B, T)
    x1[:, -1] = 1
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
    model = nBRCModel(input_dims=2,
                      hidden_dims=100,
                      output_dims=5,
                      num_layers=4,
                      full=False)
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
