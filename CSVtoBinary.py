import numpy as np
data = np.loadtxt("500000000 simulations at 0.00333 using 1.csv", delimiter=",", skiprows=1, dtype=np.float32) # using dtype=np.float32 is suggested BC CURRENT PYTORCH DEFAULTS TO 64 >:[
np.save("chunk500m1.npy", data)
