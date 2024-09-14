import numpy as np
import os
import torch

from defs import SAVED_MODELS_DIR
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

        self.steps = 0
        self.set_optimizer()

    def set_optimizer(self,lr=1,max_iter=50000,max_eval=50000,tolerance_grad=1e-5,
                    tolerance_change=0.5 * np.finfo(float).eps,history_size=50,line_search_fun='strong_wolfe'):

        self.optim = torch.optim.LBFGS(self.params,lr=lr,max_iter=max_iter,max_eval=max_eval,
                                       tolerance_grad=tolerance_grad,tolerance_change=tolerance_change,
                                       history_size=history_size,line_search_fn=line_search_fun)

        return

    def closure(self):
        '''
        Closure function required for LBFGS optimizer
        '''
        self.optim.zero_grad()
        input_data = torch.cat((self.x,self.y,self.t), dim=1).to(self.device)
        out = self.model(input_data)
        p_pred,psi_pred = out[:,0],out[:,1]
        u_pred = torch.autograd.grad(psi_pred,self.y,grad_outputs=torch.ones_like(psi_pred),create_graph=True)[0]
        v_pred = -1*torch.autograd.grad(psi_pred,self.x,grad_outputs=torch.ones_like(psi_pred),create_graph=True)[0]

        loss1 = self.supervised_loss(self.u,self.v,u_pred,v_pred)
        loss2 = self.physics_informed_loss(self.x,self.y,self.t,u_pred,v_pred,p_pred)
        loss = loss1 + loss2
        loss.backward()

        print(f"Step #{self.steps}, Total Loss: {loss}, Supervised Loss: {loss1}, Physics Loss: {loss2}, lambda1: {self.physics_informed_loss.lambda1.item()}, lambda2: {self.physics_informed_loss.lambda2.item()}",end="\r")
        self.steps += 1

        return loss

    def fit(self):
        # TODO Set up functionality for minibatching
        # This is only called once for full-batch training
        self.model.train()
        self.optim.step(self.closure)

        return

    def predict(self,x,y,t):

        x = torch.tensor(x,requires_grad=True,dtype=torch.float32,device=self.device)
        y = torch.tensor(y,requires_grad=True,dtype=torch.float32,device=self.device)
        t = torch.tensor(t,requires_grad=True,dtype=torch.float32,device=self.device)
        data = torch.cat((x,y,t), dim=1).to(self.device)

        out = self.model(data)
        p_pred,psi_pred = out[:,0],out[:,1]
        u_pred = torch.autograd.grad(psi_pred,y,grad_outputs=torch.ones_like(psi_pred),create_graph=True)[0]
        v_pred = -1*torch.autograd.grad(psi_pred,x,grad_outputs=torch.ones_like(psi_pred),create_graph=True)[0]

        return p_pred.detach().cpu().numpy(),u_pred.detach().cpu().numpy(),v_pred.detach().cpu().numpy()

    def save_model(self,fname):
        torch.save(self.model.state_dict(),os.path.join(SAVED_MODELS_DIR,fname))
        return

    def load_model(self,model_name,model_dir=SAVED_MODELS_DIR):
        self.model = torch.load(os.path.join(model_dir,model_name))
        return
