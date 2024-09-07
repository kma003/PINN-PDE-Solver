import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_8(nn.Module):

    def __init__(self,num_inputs,num_outputs,hidden_size=20):

        self.l1 = nn.Linear(num_inputs,hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size,hidden_size)
        self.l4 = nn.Linear(hidden_size,hidden_size)
        self.l5 = nn.Linear(hidden_size,hidden_size)
        self.l6 = nn.Linear(hidden_size,hidden_size)
        self.l7 = nn.Linear(hidden_size,hidden_size)
        self.l8 = nn.Linear(hidden_size,num_outputs)

    def forward(self,x):

        x = self.l1(x)
        x = self.l2(F.tanh(x))
        x = self.l3(F.tanh(x))
        x = self.l4(F.tanh(x))
        x = self.l5(F.tanh(x))
        x = self.l6(F.tanh(x))
        x = self.l7(F.tanh(x))
        x = self.l8(x)
        
        return x