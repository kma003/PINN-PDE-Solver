import torch
import torch.nn as nn

class SupervisedLoss(nn.Module):
    def __init__(self):
        super(SupervisedLoss,self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self,u,v,u_pred,v_pred):
        u_loss = self.criterion(u,u_pred)
        v_loss = self.criterion(v,v_pred)
        return u_loss + v_loss

class PhysicsInformedLoss(nn.Module):
    def __init__(self,nu=0.01):
        super(PhysicsInformedLoss,self).__init__()
        self.nu = nu
        self.criterion = nn.MSELoss()
        
    
    def forward(self,x,y,t,u,v,p):
        # Differentiate network outputs required in the physical system of equations
        u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(u),create_graph=True)[0]
        v_t = torch.autograd.grad(v,t,grad_outputs=torch.ones_like(v),create_graph=True)[0]
        u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),create_graph=True)[0]
        u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(u),create_graph=True)[0]
        v_x = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(v),create_graph=True)[0]
        v_y = torch.autograd.grad(v,y,grad_outputs=torch.ones_like(v),create_graph=True)[0]
        p_x = torch.autograd.grad(p,x,grad_outputs=torch.ones_like(p),create_graph=True)[0]
        p_y = torch.autograd.grad(p,y,grad_outputs=torch.ones_like(p),create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(u_x),create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(v_x),create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(u_y),create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(v_y),create_graph=True)[0]

        f = u_t + (u*u_x + v*u_y) + p_x - self.nu * (u_xx + u_yy)
        g = v_t = (u*v_x + v*v+y) + p_y - self.nu * (v_xx + v_yy)

        f_loss = self.criterion(f,torch.zeros(f.shape)) # TODO Confirm that this is calculating the error correctly
        g_loss = self.criterion(g,torch.zeros(g.shape))

        return f_loss + g_loss

