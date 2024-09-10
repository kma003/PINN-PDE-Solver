import numpy as np
import os
import torch
import torch.nn as nn
from scipy.io import loadmat

from defs import DATA_DIR
from networks.losses import SupervisedLoss,PhysicsInformedLoss
from networks.MLP_8 import MLP_8

class NS_PINN():
    """ Navier-Stokes solver using PINN """

    def __init__(self,x,y,t,u,v,p):
        self.x = torch.Tensor(x,dtype=torch.float32)
        self.y = torch.Tensor(y,dtype=torch.float32)
        self.t = torch.Tensor(t,dtype=torch.float32)
        self.u = torch.Tensor(u,dtype=torch.float32)
        self.v = torch.Tensor(v,dtype=torch.float32)
        self.p = torch.Tensor(p,dtype=torch.float32)

        self.model = MLP_8(num_inputs=3,num_outputs=2)
        self.supervised_loss = SupervisedLoss()
        self.physics_informed_loss = PhysicsInformedLoss()
        self.params = list(self.model.parameters()).extend(self.physics_informed_loss.parameters())
    
        self.init_optim_flag = False

    def init_optimizer(self,lr=1e-4,max_iter=20,max_eval=None,tolerance_grad=1e-7,
                    tolerance_change=1e-9,history_size=100,line_search_fun='strong_wolfe'):

        self.optim = torch.optim.LBFGS(self.params,lr=lr,max_iter=max_iter,max_eval=max_eval,
                                       tolerance_grad=tolerance_grad,tolerance_change=tolerance_change,
                                       history_size=history_size,line_search_fn=line_search_fun)
        self.init_optim_flag = True

        return

    def closure(self):
        '''
        Closure function required for LBFGS optimizer
        '''
        if self.init_optim_flag:
            self.optim.zero_grad()
            p_pred,psi_pred = self.model(self.x,self.y,self.t)
            u_pred = torch.autograd.grad(psi_pred,self.y,grad_outputs=torch.ones_like(psi_pred),create_graph=True)[0]
            v_pred = -1*torch.autograd.grad(psi_pred,self.x,grad_outputs=torch.ones_like(psi_pred),create_graph=True)[0]

            loss1 = SupervisedLoss(self.u,self.v,u_pred,v_pred)
            loss2 = PhysicsInformedLoss(self.x,self.y,self.t,u_pred,v_pred)
            loss = loss1 + loss2
            loss.backward()
        else:
            raise Exception("L-BFGS optimizer hasn't been initialzed yet")

        return loss
    
    def fit(self):
        # TODO Set up functionality for minibatching
        # This is only called once for full-batch training
        self.model.train()
        self.optim.step()
