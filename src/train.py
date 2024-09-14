import numpy as np
import os
import pickle

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

np.random.seed(0)
n_samples = 5000
idxs = np.random.choice(n_total,size=n_samples,replace=False)
x_train = x_repeated[idxs]
y_train = y_repeated[idxs]
t_train = t_repeated[idxs]
u_train = u_flattened[idxs]
v_train = v_flattened[idxs]
p_train = p_flattened[idxs]

# Set up PINN and fit data
pinn = NS_PINN(x=x_train,y=y_train,t=t_train,u=u_train,v=v_train,p=p_train)

# Fit data
pinn.fit()
pinn.save_model('model.pt')

p_pred,u_pred,v_pred = pinn.predict(x_train,y_train,t_train)

def preprocess(x,y,t):
    x_processed = (x - np.min(x)) / (np.max(x) - np.min(x))
    y_processed = (y - np.min(y)) / (np.max(y) - np.min(y))
    t_processed = (y - np.min(t)) / (np.max(t) - np.min(t))

    return x_processed,y_processed,t_processed

def evaluate():
    pass