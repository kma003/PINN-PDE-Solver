import numpy as np
import os
import pickle
import torch

from src.ns_pinn import NS_PINN
from src.defs import DATA_DIR

with open(os.path.join(DATA_DIR,'cylinder_wake_data.pkl'), 'rb') as f:
    data = pickle.load(f)

t = data['t'] # Shape T x 1
x = data['x'] # Shape N x 1
y = data['y'] # Shape N x 1
U = data['U'] # Shape N x T
V = data['V'] # Shape N x T
P = data['P'] # Shape N x T

n_time = t.shape[0]
n_pos = x.shape[0]
n_total = n_time * n_pos

# Repeat vector of positional coordinates for each point in time
x_repeated = np.expand_dims(np.tile(x,n_time),axis=1) # Shape (N*T) x 1
y_repeated = np.expand_dims(np.tile(y,n_time),axis=1) # Shape (N*T) x 1

# Repeat each timepoint element for all positions 
t_repeated = np.repeat(t,n_pos,axis=0) # Shape (N*T) x 1

# Flatten matrices into vectors
u_flattened = np.reshape(U,(n_total,1)) # Shape (N*T) x 1
v_flattened = np.reshape(V,(n_total,1)) # Shape (N*T) x 1
p_flattened = np.reshape(P,(n_total,1)) # Shape (N*T) x 1

# Set up PINN and fit data
pinn = NS_PINN(x=x_repeated,y=y_repeated,t=t_repeated,
               u=u_flattened,v=v_flattened,p=p_flattened)

# Fit data
pinn.fit()

