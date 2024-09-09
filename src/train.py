import numpy as np
import os
import torch
import torch.nn as nn
from scipy.io import loadmat

from defs import DATA_DIR
from networks.losses import SupervisedLoss,PhysicsInformedLoss
from networks.MLP_8 import MLP_8



def train(learning_rate=1e-4,num_epochs=100):

    data_path = os.path.join(DATA_DIR,"cylinder_wake.mat")
    data = loadmat(data_path)

    lr = learning_rate
    model = MLP_8(num_inputs=3,num_outputs=2)
    optimizer = torch.optim.LBFGS(model.parameters())
    criterion1 = SupervisedLoss()
    criterion2 = PhysicsInformedLoss()

    for epoch in range(num_epochs):
        model.train()
        x=y=t=u=v=p = None # TODO Set up input data

        optimizer.zero_grad()
        p_pred,psi = model(torch.Tensor([x,y,t]))

        # Differentiate for predictions, calculate losses
        u_pred = torch.autograd.grad(psi,y,grad_outputs=torch.ones_like(psi),create_graph=True)[0]
        v_pred = -1*torch.autograd.grad(psi,u,grad_outputs=torch.ones_like(psi),create_graph=True)[0]
        loss1 = SupervisedLoss(u,v,u_pred,v_pred)
        loss2 = PhysicsInformedLoss(x,y,t,u_pred,v_pred)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        pass

    return


if __name__ == "__main__":

    train()
