import torch 
import torch.nn.functional as F
from torch.linalg import lstsq
import torch.nn as nn
import numpy as np 

def fourier_best_approx(W, x, y):
    """
    Given frequencies w of a fourier type series, computes the coefficients such that
    the euclidean norm of f(x_i)-y_i is minimized.

    W (torch.tensor) : gives a tensor of n frequencies, each of size (1, l), all in all
        a tensor of size (n, l)
    
    x (torch.tensor) : the m datapoints corresponding to the labels y_i, each datapoint x_i is of
        dimension (1, l), such that x is (m, l)
    
    y (torch.tensor) : the m labels, dimension (1,m)
    """
    W = W.t()
    z = x.matmul(W)
    A_cos = z.cos()
    A_sin = z.sin()
    A = torch.cat((A_cos, A_sin), 1)
    coeffs = torch.linalg.lstsq(A,y, driver='gelsd').solution
    return coeffs

class Fourier_model(nn.Module):
    def __init__(self, frequencies):
        super().__init__()
        self.W = frequencies.t()
        self.input_dim = frequencies.shape[0]
        self.linear = nn.Linear(self.input_dim, 1).double()

    def forward(self, x):
        z = x.matmul(self.W)
        sin_z = self.linear(z.sin())
        cos_z = self.linear(z.cos())
        output = sin_z + cos_z
        return output


