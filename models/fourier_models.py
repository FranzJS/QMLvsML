import torch 
import torch.nn.functional as F
from torch.linalg import lstsq
import torch.nn as nn
import numpy as np 


class Fourier_model(nn.Module):
    def __init__(self, frequencies, output_dim=1):
        super().__init__()
        self.W = frequencies.t()
        self.input_dim = frequencies.shape[0]
        self.output_dim = output_dim
        self.linear_sin = nn.Linear(self.input_dim, self.output_dim, bias=False).double()
        self.linear_cos = nn.Linear(self.input_dim, self.output_dim, bias=False).double()

    def forward(self, x):
        z = x.matmul(self.W)
        sin_z = self.linear_sin(z.sin())
        cos_z = self.linear_cos(z.cos())
        output = sin_z + cos_z
        return output
