import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# sometime ill have to make it more useable with chunks
data = np.load("chunk200m1.npy").astype(np.float32)

state_mean = data[:, :6].mean(axis=0, dtype=np.float32)
state_std = data[:, :6].std(axis=0, dtype=np.float32)
data[:, :6] = (data[:, :6] - state_mean) / state_std

target_mean = data[:, 6:].mean(axis=0, dtype=np.float32)
target_std = data[:, 6:].std(axis=0, dtype=np.float32)
data[:, 6:] = (data[:, 6:] - target_mean) / target_std

class Predictor(nn.Module):
    def __init__(self, input_dim=6, output_dim=2,
                 hidden_dims=[350, 350, 350, 350, 350, 350, 350, 350], # this seems just way to big BUT after trying many networks of many widths and depth it appears to be the most optimal. best guess is chaotic systems really benifit from skipped conns and depth allows for this since the entire point of chaos is sensitivity to early input.
                 skip_connections=[(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8)],
                 norm = "None", #in theory Bnorm should be optimal but in my sweeps it always performed worse, and Lnorm took more than 5x as long to converge
                 ):
        super().__init__()
        if skip_connections is None:
            skip_connections = []

        self.hidden_dims = hidden_dims
        self.skip_connections = skip_connections
        self.norm = norm
        # Track layer input sizes
        layer_input_dims = [input_dim]
        for hid in hidden_dims:
            layer_input_dims.append(hid)

        self.layers = nn.ModuleList()
        for i, hid_dim in enumerate(hidden_dims):
            skip_to = [src for src, dst in skip_connections if dst == i+1]
            extra_dim = sum(layer_input_dims[src] for src in skip_to)
            in_dim = layer_input_dims[i] + extra_dim

            if self.norm == "Layer":
                norm = nn.LayerNorm(hid_dim)
            elif self.norm == "Batch":
                norm = nn.BatchNorm1d(hid_dim)
            else: norm = nn.Identity() # effectively 'None'. but since you cant pass 'None' pytorch only uses the first activation function called and ignores any others following.

            linear = nn.Linear(in_dim, hid_dim)
            nn.init.normal_(linear.weight, mean=0.0, std=0.04),
            nn.init.zeros_(linear.bias),

            layer = nn.Sequential(
                linear,
                nn.GELU(), # GELU outperforms all activation functions by a huge margin
                norm #
            )
            self.layers.append(layer)

        linear = nn.Linear(hidden_dims[-1], output_dim)
        nn.init.normal_(linear.weight, mean=0.0, std=0.015),
        nn.init.zeros_(linear.bias),
        self.output_head = linear

    def forward(self, x):
        outputs = [x]
        out = x
        for i, layer in enumerate(self.layers):
            skip_to = [src for src, dst in self.skip_connections if dst == i+1]
            if skip_to:
                concat_inputs = [outputs[src] for src in skip_to]
                out = torch.cat([out] + concat_inputs, dim=1)
            out = layer(out)
            outputs.append(out)
        return self.output_head(out)


model = Predictor().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0015)
criterion = nn.MSELoss()

batch_size = 16_000
num_epochs = 15

dataset_size = len(data)
dataset = torch.from_numpy(data)
"""loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,   # key!
    num_workers=5      # optional, parallel loading
)""" # python data shenanigans i mentioned, will remove once i verify they will be of no help

loggingRate = 2 # second(s)

for epoch in range(num_epochs):
    print(1)
    indices = np.random.permutation(dataset_size)
    #batch_losses = []

    nextTime = time.time() + loggingRate

    for start in range(0, dataset_size, batch_size):
        end = min(start + batch_size, dataset_size)
        idx = indices[start:end]
        batch = dataset[idx]
        states = batch[:, :6].to(device)
        targets = batch[:, 6:].to(device)

        optimizer.zero_grad()
        preds = model(states)
        loss = criterion(preds, targets)
        loss_item = loss.item()

        if time.time() > nextTime:
            print(loss_item) # once i am sure of my finished models efficacy ill use wandb for overnight sweeps, until then im lazy and just printing
            nextTime += loggingRate

        loss.backward()
        optimizer.step()

PATH = "900kmodel.pth"
torch.save(model.state_dict(), PATH) 

"""for states_targets in loader:
        states_targets = states_targets.to(device, non_blocking=True)
        states, targets = states_targets[:, :6], states_targets[:, 6:]""" # python data shenanigans i mentioned, will remove once i verify they will be of no help
