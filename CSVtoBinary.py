import numpy as np
import json

with open("config.json") as f:
    cfg = json.load(f)

SIMS = cfg["SIMS"]
STEPS = cfg["STEPS"]
SEED = cfg["SEED"]

FILE_NAME = f"{SIMS} simulations at {STEPS} per second using {SEED} seed"

try:
    data = np.loadtxt(f"{FILE_NAME}.csv", delimiter=",", skiprows=1, dtype=np.float32)
except Exception as e:
    print("failed to load data", e)

input_mean = data[:, :7].mean(axis=0, dtype=np.float32)
input_std = data[:, :7].std(axis=0, dtype=np.float32)
target_mean = data[:, 7:].mean(axis=0, dtype=np.float32)
target_std = data[:, 7:].std(axis=0, dtype=np.float32)

try:
    np.savez(f"{FILE_NAME} stats.npz", input_mean=input_mean, input_std=input_std, target_mean=target_mean, target_std=target_std)
except Exception as e:
    print("failed to save stats.npz", e)

del input_mean, input_std, target_mean, target_std

try:
    np.save(f"{FILE_NAME} data.npy", data)
except Exception as e:
    print("failed to save data", e)
