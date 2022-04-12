#!/usr/bin/env python
#coding: utf-8

#SBATCH --job-name=pennylane_reg_00                
#SBATCH --ntasks=8                     # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=0-14:00:00              # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem=8G             
#SBATCH --output=job1_%j.out           
#SBATCH --error=job1_%j.err            
#SBATCH --mail-type=END                
#SBATCH --mail-user=schreibef98@zedat.fu-berlin.de   

# +
import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader
from pennylane import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f'Using {device} device')
# -

import sys
sys.path.insert(0, "../")
import utils.utils as utils
import models.fourier_models as fm
import models.quantum_models as qm

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# +
# generate a suitable regression task
n_samples = 200
n_features = 3
n_informative = 3
n_targets = 1
noise = 0.0
random_state = 42
X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_targets=n_targets, noise=noise, random_state=random_state)
X, y = torch.from_numpy(X), torch.from_numpy(y)
# Scale data to interval [-pi/2, pi/2]
X_scaled = utils.data_scaler(X, interval=(-torch.pi/2, torch.pi/2))

# Split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2)

# torch dataloaders
train_data_list = []
for i in range(len(X_train)):
    data_point = (X_train[i], y_train[i])
    train_data_list.append(data_point)

test_data_list = []
for i in range(len(X_test)):
    data_point = (X_test[i], y_test[i])
    test_data_list.append(data_point)
    
train_dataloader = DataLoader(train_data_list, batch_size=200, shuffle=True)
test_dataloader = DataLoader(test_data_list, batch_size=200, shuffle=True)
# -

n_qubits = n_features
model = qm.QuantumRegressionModel(n_qubits, n_layers=2, n_trainable_block_layers=1)
#model.load_state_dict(torch.load("../data/pennylane_regression_minima_search_02/run_0_initial_model.pt"))
model.load_state_dict(torch.load("../data/pennylane_regression_minima_search_04/run_0_model_epoch_50.pt"))
model.to(device)
print(list(model.parameters()))


# +
def train(dataloader, model, loss_fn, optimizer, printing=False):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred.flatten(), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            if printing == True:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                for param_group in optimizer.param_groups:
                    print("lr: ", param_group['lr'])
        return loss

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred.flatten(), y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


# -

def ctrain(dataloader, model, loss_fn, optimizer, printing=False):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        def closure():
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output.flatten(), y)
            loss.backward()
            return loss
        
        optimizer.step(closure)

        if batch % 100 == 0:
            loss = loss_fn(model(X).flatten(), y)
            loss, current = loss.item(), batch * len(X)
            if printing == True:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                for param_group in optimizer.param_groups:
                    print("lr: ", param_group['lr'])
        return loss


i = 0

    
n_qubits = n_features
model = qm.QuantumRegressionModel(n_qubits, n_layers=2, n_trainable_block_layers=1)
#model.load_state_dict(torch.load(load_path+"initial_model.pt"))
model.to(device)
#print(list(model.parameters()))
loss_fn = nn.MSELoss(reduction='mean') # equiv. to torch.linalg.norm(input-target)**2
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5)

NN_loss = []
NN_test_loss = []

save_path = "../data/pennylane_regression_01/run_{}.1_".format(i)

epochs = 50
for t in tqdm(range(epochs)):
    # print(f"Epoch {t+1}\n-------------------------------")
    NN_loss.append(ctrain(train_dataloader, model, loss_fn, optimizer, printing=True))
    NN_test_loss.append(test(test_dataloader, model, loss_fn))    
    if t % 10 == 0:
        #pass
        torch.save(model.state_dict(), save_path+"model_epoch_{}.pt".format(t))
    if np.isnan(NN_loss[-1]) == True:
        break
            
    np.save(save_path+"NN_loss.npy", np.array(NN_loss))
    np.save(save_path+"NN_test_loss.npy", np.array(NN_test_loss))
    print("Done!")
