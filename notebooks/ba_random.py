#!/usr/bin/env python
#coding: utf-8

#SBATCH --job-name=ba_reg_1                
#SBATCH --ntasks=1                     # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=0-14:00:00              # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem=8G   
#SBATCH --array=1-1       
#SBATCH --output=/scratch/schreibef98/QMLvsML/data/ba_random/job%A_%a.out           
#SBATCH --error=/scratch/schreibef98/QMLvsML/data/ba_random/job%A_&a.err            
#SBATCH --mail-type=END                
#SBATCH --mail-user=schreibef98@zedat.fu-berlin.de


import numpy as np

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch import nn
device = 'cpu'
print(f'Using {device} device')

import sys
sys.path.insert(0, "../")
import utils.utils as utils
import models.fourier_models as fm
import models.quantum_models as qm

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


# generate a suitable regression task
n_samples = 500
n_dim = 4
n_layers = 2
n_trainable_block_layers = 3

target_model = qm.QuantumRegressionModel(n_dim, n_layers=n_layers, n_trainable_block_layers=n_trainable_block_layers)
rng = np.random.default_rng()
X = torch.from_numpy(rng.random((n_samples, n_dim)))
with torch.no_grad():
    y = target_model(X).flatten()
X_scaled = utils.data_scaler(X, interval=(-torch.pi/2, torch.pi/2))

# Split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)


dim = X_scaled[0].shape[0]

ba_loss = []
ba_test_loss = []
for max_freq in range(1, 3+1):
    W = utils.freq_generator(max_freq, dim)
    ba_coeffs = utils.fourier_best_approx(W, X_train, y_train)
    ba_loss.append(utils.loss(W, ba_coeffs, X_train, y_train).item())
    ba_test_loss.append(utils.loss(W, ba_coeffs, X_test, y_test).item())
    print("max_freq: ", max_freq)
    print("training_loss: ",utils.loss(W, ba_coeffs, X_train, y_train).item())
    print("test loss: ",utils.loss(W, ba_coeffs, X_test, y_test).item())
    print("parameters: ", len(ba_coeffs))

save_path = '/scratch/schreibef98/QMLvsML/data/ba_random/'

"""
np.save(save_path+"ba_loss.npy", np.array(ba_loss))
np.save(save_path+"ba_test_loss.npy", np.array(ba_test_loss))
"""