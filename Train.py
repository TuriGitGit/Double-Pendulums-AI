import time
import os
import json
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.init as init

from heavyball import ForeachMuon

import wandb

class Predictor(nn.Module):
    def __init__(self, input_dim=7, output_dim=2,
                 hidden_dims=[400]*8,
                 skip_connections=[(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8)],
                 lr=0.0014, clip=8.0, batch_size=3750, init_std_L=0.037, init_std_O=0.0046
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
                nn.GELU(),
            )
            self.layers.append(layer)

        linear = nn.Linear(hidden_dims[-1], output_dim)
        init.normal_(linear.weight, mean=0.0, std=self.init_std_O)
        init.zeros_(linear.bias)
        self.output_head = linear

        self.optimizer = ForeachMuon(self.parameters(), lr=lr)

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
      "distribution": "log_uniform_values",
      "min": 0.0001,
      "max": 0.01
    },
    "clip": {
      "distribution": "uniform",
      "min": 0.5,
      "max": 8.0
    },
    "batch_size": {
      "distribution": "q_uniform",
      "min": 1_000,
      "max": 50_000,
      "q": 100
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

def sweep():
    with wandb.init() as run:
        config = wandb.config
        agent = Predictor(
            lr=config.lr, clip=config.clip, batch_size=config.batch_size, init_std_L=config.init_std_L, init_std_O=config.init_std_O
        ).to(device)

        recent_avg_losses = deque(maxlen=500)
        for epoch in range(1):
            start_time = time.time()
            end_time = 600 # seconds
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

                if time.time() >= start_time + end_time:
                    wandb.finish(exit_code=0)
                    return

                loss.backward()
                utils.clip_grad_norm_(agent.parameters(), agent.clip)
                agent.optimizer.step()

def train():
    agent = Predictor().to(device)
    #agent.load_state_dict(torch.load(MODEL, map_location=device)) TODO: allow easy chunked training

    for epoch in range(10):
        epoch_loss = 0.0
        indices = np.random.permutation(dataset_size)

        for i, start in enumerate(range(0, dataset_size - agent.batch_size, agent.batch_size)):
            end = min(start + agent.batch_size, dataset_size)
            idx = indices[start:end]
            batch = dataset[idx]
            inputs = batch[:, :7].to(device)
            targets = batch[:, 7:].to(device)

            agent.optimizer.zero_grad()
            preds = agent(inputs)
            loss = criterion(preds, targets)
            epoch_loss += loss.item()

            loss.backward()
            utils.clip_grad_norm_(agent.parameters(), agent.clip)
            agent.optimizer.step()

        epoch_loss /= (i+1)
        print(f"{epoch_loss:.6f}")

        os.makedirs("Models", exist_ok=True)

        torch.save(agent.state_dict(), os.path.join("Models",
            f"[{agent.hidden_dims[0]}]*{len(agent.hidden_dims)} size NN trained on {FILE_NAME} data for {epoch+1} epochs.pth")
        )

if __name__ == "__main__":
    with open("config.json") as f:
        cfg = json.load(f)

    SIMS = cfg["SIMS"]
    STEPS = cfg["STEPS"]
    SEED = cfg["SEED"]

    FILE_NAME = f"{SIMS} simulations at {STEPS} per second using {SEED} seed"

    STATS = f"{FILE_NAME} stats.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    data = np.load(f"{FILE_NAME} data.npy")
    stats = np.load(STATS)
    input_mean, input_std = stats['input_mean'], stats['input_std']
    target_mean, target_std = stats['target_mean'], stats['target_std']
    del stats

    np.subtract(data[:, :7], input_mean, out=data[:, :7])
    np.divide(data[:, :7], input_std, out=data[:, :7])
    # slower, but inplace uses less memory, since this part of the code uses the most memory and takes up negligeable time, its worth it.
    np.subtract(data[:, 7:], target_mean, out=data[:, 7:])
    np.divide(data[:, 7:], target_std, out=data[:, 7:])


    dataset_size = len(data)
    dataset = torch.from_numpy(data)

    criterion = nn.MSELoss()

    train()

    #sweep_id = wandb.sweep(sweep_config, project="ProjectName") 
    #wandb.agent(sweep_id, sweep, count=100) # use these if you want to run a sweep
