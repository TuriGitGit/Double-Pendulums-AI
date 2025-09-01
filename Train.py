import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# sometime ill have to make it more useable with chunks
data = np.load("chunk500m1.npy").astype(np.float32)

input_mean = data[:, :7].mean(axis=0, dtype=np.float32)
input_std = data[:, :7].std(axis=0, dtype=np.float32)
data[:, :7] = (data[:, :7] - input_mean) / input_std

target_mean = data[:, 7:].mean(axis=0, dtype=np.float32)
target_std = data[:, 7:].std(axis=0, dtype=np.float32)
data[:, 7:] = (data[:, 7:] - target_mean) / target_std

class Predictor(nn.Module):
    def __init__(self, input_dim=7, output_dim=2,
                 hidden_dims=[350, 350, 350, 350, 350, 350, 350, 350],
                 skip_connections=[(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8)],
                 norm = "None", # in theory Bnorm should be optimal but in my sweeps it always performed worse, and Lnorm took more than 5x as long to converge
                 ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.skip_connections = skip_connections
        self.norm = norm

        self.film_generators = nn.ModuleList([nn.Linear(1, 2 * hid) for hid in hidden_dims])
        for fg, hid in zip(self.film_generators, hidden_dims):
            nn.init.ones_(fg.weight[:, :hid])
            nn.init.zeros_(fg.weight[:, hid:])
            nn.init.zeros_(fg.bias)

        layer_input_dims = [input_dim]
        for hid in hidden_dims:
            layer_input_dims.append(hid)

        self.layers = nn.ModuleList()
        for i, hid_dim in enumerate(hidden_dims):
            skip_to = [src for src, dst in skip_connections if dst == i+1]
            extra_dim = sum(layer_input_dims[src] for src in skip_to)
            in_dim = layer_input_dims[i] + extra_dim

            if self.norm == "Layer": norm = nn.LayerNorm(hid_dim)
            elif self.norm == "Batch": norm = nn.BatchNorm1d(hid_dim)
            else: norm = nn.Identity() # effectively 'None'. but since you cant pass 'None' pytorch only uses the first activation function called and ignores any others following.

            linear = nn.Linear(in_dim, hid_dim)
            nn.init.normal_(linear.weight, mean=0.0, std=0.05)
            nn.init.zeros_(linear.bias)

            layer = nn.Sequential(
                linear,
                nn.GELU(),
                norm
            )
            self.layers.append(layer)

        linear = nn.Linear(hidden_dims[-1], output_dim)
        nn.init.normal_(linear.weight, mean=0.0, std=0.015)
        nn.init.zeros_(linear.bias)
        self.output_head = linear

    def forward(self, x):
        temporal_factor = x[:, -1].unsqueeze(1)
        outputs = [x]
        out = x
        for i, layer in enumerate(self.layers):
            skip_to = [src for src, dst in self.skip_connections if dst == i+1]
            if skip_to:
                concat_inputs = [outputs[src] for src in skip_to]
                out = torch.cat([out] + concat_inputs, dim=1)
            out = layer(out)

            modulation_scale = 0.05
            gamma_beta = self.film_generators[i](temporal_factor)
            gamma, beta = gamma_beta.chunk(2, dim=1)
            out = (1 + modulation_scale * (gamma - 1)) * out + modulation_scale * beta

            outputs.append(out)
        return self.output_head(out)


model = Predictor().to(device)
optimizer = torch.optim.Adam([
    {"params": model.layers.parameters(), "lr": 0.0015}, # weight decay may be useful now, will have to sweep.
    {"params": model.film_generators.parameters(), "lr": 0.00008}
])
criterion = nn.MSELoss()

batch_size = 16_000
num_epochs = 20

dataset_size = len(data)
dataset = torch.from_numpy(data)

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
        inputs = batch[:, :7].to(device)
        targets = batch[:, 7:].to(device)

        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss_item = loss.item()

        if time.time() > nextTime:
            print(loss_item) # once i am sure of my finished models efficacy ill use wandb for overnight sweeps, until then im lazy and just printing
            nextTime += loggingRate

        loss.backward()
        optimizer.step()

PATH = "900k_model.pth"
torch.save(model.state_dict(), PATH)
