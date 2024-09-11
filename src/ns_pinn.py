import numpy as np
import torch

from networks.losses import SupervisedLoss,PhysicsInformedLoss
from networks.MLP_8 import MLP_8

class NS_PINN():
    """ Navier-Stokes solver using PINN """

    def __init__(self,x,y,t,u,v,p):
        self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x = torch.tensor(x,requires_grad=True,dtype=torch.float32,device=self.device)
        self.y = torch.tensor(y,requires_grad=True,dtype=torch.float32,device=self.device)
        self.t = torch.tensor(t,requires_grad=True,dtype=torch.float32,device=self.device)
        self.u = torch.tensor(u,dtype=torch.float32,device=self.device)
        self.v = torch.tensor(v,dtype=torch.float32,device=self.device)
        self.p = torch.tensor(p,dtype=torch.float32,device=self.device)

        self.model = MLP_8(num_inputs=3,num_outputs=2).to(self.device)
        self.supervised_loss = SupervisedLoss()
        self.physics_informed_loss = PhysicsInformedLoss()
        self.params = list(self.model.parameters())
        self.params.extend(list(self.physics_informed_loss.parameters()))

        self.init_optimizer()

    def init_optimizer(self,lr=1e-4,max_iter=20,max_eval=None,tolerance_grad=1e-7,
                    tolerance_change=1e-9,history_size=100,line_search_fun='strong_wolfe'):

        self.optim = torch.optim.LBFGS(self.params,lr=lr,max_iter=max_iter,max_eval=max_eval,
                                       tolerance_grad=tolerance_grad,tolerance_change=tolerance_change,
                                       history_size=history_size,line_search_fn=line_search_fun)

        return

    def closure(self):
        '''
        Closure function required for LBFGS optimizer
        '''
        self.optim.zero_grad()
        input_data = torch.cat((self.x,self.y,self.t), dim=1)
        out = self.model(input_data)
        p_pred,psi_pred = out[:,0],out[:,1]
        u_pred = torch.autograd.grad(psi_pred,self.y,grad_outputs=torch.ones_like(psi_pred),create_graph=True)[0]
        v_pred = -1*torch.autograd.grad(psi_pred,self.x,grad_outputs=torch.ones_like(psi_pred),create_graph=True)[0]

        loss1 = self.supervised_loss(self.u,self.v,u_pred,v_pred)
        loss2 = self.physics_informed_loss(self.x,self.y,self.t,u_pred,v_pred,p_pred)
        loss = loss1 + loss2
        loss.backward()


        return loss
    
    def fit(self):
        # TODO Set up functionality for minibatching
        # This is only called once for full-batch training
        self.model.train()
        self.optim.step(self.closure)
