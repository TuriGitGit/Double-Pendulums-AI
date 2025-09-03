import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
import torch.nn.init as init

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data = np.load("chunk100m1.npy").astype(np.float32)

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
                 lr=0.0012, clip=7.6, batch_size=2716, init_std_L=0.039, init_std_O=0.0046
                 ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.skip_connections = skip_connections
        self.lr = lr
        self.clip = clip
        self.batch_size = batch_size
        self.init_std_L = init_std_L
        self.init_std_O = init_std_O

        layer_input_dims = [input_dim]
        for hid in hidden_dims:
            layer_input_dims.append(hid)

        self.layers = nn.ModuleList()
        for i, hid_dim in enumerate(hidden_dims):
            skip_to = [src for src, dst in skip_connections if dst == i+1]
            extra_dim = sum(layer_input_dims[src] for src in skip_to)
            in_dim = layer_input_dims[i] + extra_dim

            linear = nn.Linear(in_dim, hid_dim)
            init.normal_(linear.weight, mean=0.0, std=self.init_std_L)
            init.zeros_(linear.bias)

            layer = nn.Sequential(
                linear,
                nn.GELU(), # no need to sweep activation, GELU dominates.
            )
            self.layers.append(layer)

        linear = nn.Linear(hidden_dims[-1], output_dim)
        init.normal_(linear.weight, mean=0.0, std=self.init_std_O)
        init.zeros_(linear.bias)
        self.output_head = linear

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

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


sweep_config = {
  "method": "bayes",
  "metric": {
    "name": "recent_avg_loss",
    "goal": "minimize"
  },
  "parameters": {
      "lr": {
      "distribution": "uniform",
      "min": 0.0001,
      "max": 0.01
    },
    "clip": {
      "distribution": "uniform",
      "min": 0.5,
      "max": 8.0
    },
    "batch_size": {
      "distribution": "int_uniform",
      "min": 1_000,
      "max": 50_000
    },
    "init_std_L": {
      "distribution": "uniform",
      "min": 0.01,
      "max": 0.05
    },
    "init_std_O": {
      "distribution": "uniform",
      "min": 0.004,
      "max": 0.02
    }
  }
}

dataset_size = len(data)
dataset = torch.from_numpy(data)
criterion = nn.MSELoss()
def sweep():
    with wandb.init() as run:
        config = wandb.config
        agent = Predictor(
            lr=config.lr, clip=config.clip, batch_size=config.batch_size, init_std_L=config.init_std_L, init_std_O=config.init_std_O
        ).to(device)
        recent_avg_losses = deque(maxlen=500)

        for epoch in range(2):
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size - agent.batch_size, agent.batch_size):
                end = min(start + agent.batch_size, dataset_size)
                idx = indices[start:end]
                batch = dataset[idx]
                inputs = batch[:, :7].to(device)
                targets = batch[:, 7:].to(device)

                agent.optimizer.zero_grad()
                preds = agent(inputs)
                loss = criterion(preds, targets)
                loss_item = loss.item()
                recent_avg_losses.append(loss_item)
                recent_avg_loss = sum(recent_avg_losses) / len(recent_avg_losses) if len(recent_avg_losses) >= 500 else 1

                wandb.log({"loss": loss_item, "recent_avg_loss": recent_avg_loss})

                loss.backward()
                utils.clip_grad_norm_(agent.parameters(), agent.clip)
                agent.optimizer.step()

def train():
    agent = Predictor().to(device)

    for epoch in range(50):
        indices = np.random.permutation(dataset_size)

        for start in range(0, dataset_size - agent.batch_size, agent.batch_size):
            end = min(start + agent.batch_size, dataset_size)
            idx = indices[start:end]
            batch = dataset[idx]
            inputs = batch[:, :7].to(device)
            targets = batch[:, 7:].to(device)

            agent.optimizer.zero_grad()
            preds = agent(inputs)
            loss = criterion(preds, targets)

            loss.backward()
            utils.clip_grad_norm_(agent.parameters(), agent.clip)
            agent.optimizer.step()

    loss_item = loss.item()
    print(loss_item)
    PATH = f"{epoch+1}_{loss_item:.6f}_model.pth"
    torch.save(agent.state_dict(), PATH)


if __name__ == "__main__":
    train()

    #sweep_id = wandb.sweep(sweep_config, project="DoublePends") 
    #wandb.agent(sweep_id, sweep, count=100) # use these if you want to run a sweep
