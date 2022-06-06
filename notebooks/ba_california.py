#!/usr/bin/env python
#coding: utf-8

#SBATCH --job-name=ba_reg_1                
#SBATCH --ntasks=1                     # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=0-14:00:00              # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem=8G   
#SBATCH --array=1-1       
#SBATCH --output=/scratch/schreibef98/QMLvsML/data/ba_california/job%A_%a.out           
#SBATCH --error=/scratch/schreibef98/QMLvsML/data/ba_california/job%A_&a.err            
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

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


# generate a suitable regression task
data = fetch_california_housing()
X = torch.from_numpy(data.data)
y = torch.from_numpy(data.target)

X_scaled = utils.data_scaler(X, interval=(-torch.pi/2, torch.pi/2))

# Split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.9, random_state=42)


dim = X_scaled[0].shape[0]

ba_loss = []
ba_test_loss = []
for max_freq in range(1, 2+1):
    W = utils.freq_generator(max_freq, dim)
    ba_coeffs = utils.fourier_best_approx(W, X_train, y_train)
    ba_loss.append(utils.loss(W, ba_coeffs, X_train, y_train).item())
    #ba_test_loss.append(utils.loss(W, ba_coeffs, X_test, y_test).item())
    print("max_freq: ", max_freq)
    print("training_loss: ",utils.loss(W, ba_coeffs, X_train, y_train).item())
    print("test loss: ",utils.loss(W, ba_coeffs, X_test, y_test).item())
    print("parameters: ", len(ba_coeffs))

save_path = '/scratch/schreibef98/QMLvsML/data/ba_california/'

"""
np.save(save_path+"ba_loss.npy", np.array(ba_loss))
np.save(save_path+"ba_test_loss.npy", np.array(ba_test_loss))
"""