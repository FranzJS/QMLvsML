import torch 
import torch.nn.functional as F
from torch.linalg import lstsq
import torch.nn as nn
import numpy as np 


class Fourier_model(nn.Module):
    def __init__(self, frequencies, output_dim=1):
        super().__init__() # initialize the nn.Module class
        self.W = frequencies.t() # transpose the matrix of frequency vectors
        self.input_dim = frequencies.shape[0] # length of one frequency vector (== length of input data)
        self.output_dim = output_dim
        self.linear_sin = nn.Linear(self.input_dim, self.output_dim, bias=False).double() # weights for the sin parts
        self.linear_cos = nn.Linear(self.input_dim, self.output_dim, bias=False).double() # weights for the cos parts

    # forward pass in the NN, always done by overwriting the forward function in pytorch.
    def forward(self, x):
        z = x.matmul(self.W)
        sin_z = self.linear_sin(z.sin())
        cos_z = self.linear_cos(z.cos())
        output = sin_z + cos_z
        return output
