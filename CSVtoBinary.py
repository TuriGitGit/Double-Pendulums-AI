import numpy as np

try:
    data = np.loadtxt("150000000 simulations at 0.00333 using 1.csv", delimiter=",", skiprows=1, dtype=np.float32)
except Exception as e:
    print("failed to load data", e)

input_mean = data[:, :7].mean(axis=0, dtype=np.float32)
input_std = data[:, :7].std(axis=0, dtype=np.float32)
target_mean = data[:, 7:].mean(axis=0, dtype=np.float32)
target_std = data[:, 7:].std(axis=0, dtype=np.float32)

try:
    np.savez("150m stats.npz", input_mean=input_mean, input_std=input_std, target_mean=target_mean, target_std=target_std)
except Exception as e:
    print("failed to save stats.npz", e)

del input_mean, input_std, target_mean, target_std

try:
    np.save("150m data.npy", data)
except Exception as e:
    print("failed to save data", e)
