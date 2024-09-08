import numpy as np
import os
import torch
import torch.nn as nn
from scipy.io import loadmat

from defs import DATA_DIR
from networks.MLP_8 import MLP_8

data_path = os.path.join(DATA_DIR,"cylinder_wake.mat")
data = loadmat(data_path)

net = MLP_8(num_inputs=3,num_outputs=2)

def train():
    pass

class SupervisedLoss(nn.Module):
    def __init__(self):
        super(SupervisedLoss,self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self,u,v,u_pred,v_pred):
        u_loss = self.criterion(u,u_pred)
        v_loss = self.criterion(v,v_pred)
        return u_loss + v_loss

class PhysicsInformedLoss(nn.Module):
    pass