import numpy as np
data = np.loadtxt("200000000 simulations at 0.00235 for 2.00 using 1.csv", delimiter=",", skiprows=1, dtype=np.float32) # using dtype=np.float32 is suggested BC CURRENT PYTORCH DEFAULTS TO 64 >:[
np.save("chunk200m1.npy", data)
