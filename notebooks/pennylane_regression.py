# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader
from pennylane import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
model = qm.QuantumRegressionModel(n_qubits)


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


# +
loss_fn = nn.MSELoss(reduction='mean') # equiv. to torch.linalg.norm(input-target)**2
optimizer = torch.optim.Adam(model.parameters(), lr=0.8)
train(train_dataloader, model, loss_fn, optimizer, printing=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.2)

epochs = 100
for t in tqdm(range(epochs)):
    # print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, printing=True)
    # test(test_dataloader, model, loss_fn)
    # scheduler.step()
print("Done!")
print(train(train_dataloader, model, loss_fn, optimizer, printing=True))
print(test(test_dataloader, model, loss_fn))

# +
# generate frequencies
max_freq = 2
dim = X_scaled[0].shape[0]

W = utils.freq_generator(max_freq, dim)

# compute best approximation
ba_coeffs = utils.fourier_best_approx(W, X_train, y_train)

print("training_loss: ",utils.loss(W, ba_coeffs, X_train, y_train))
print("test loss: ",utils.loss(W, ba_coeffs, X_test, y_test))
# -

model(X_train[0])

y_train[0]

utils.loss(W, ba_coeffs, X_train[0:2], y_train[0:2])

epochs = 100
for t in tqdm(range(epochs)):
    # print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, printing=True)
    # test(test_dataloader, model, loss_fn)
    # scheduler.step()
print("Done!")
print(train(train_dataloader, model, loss_fn, optimizer, printing=True))
print(test(test_dataloader, model, loss_fn))

# +
W = utils.freq_generator(max_freq, dim).to(device)
model = fm.Fourier_model(W)
model.to(device)
loss_fn = nn.MSELoss(reduction='mean') # equiv. to torch.linalg.norm(input-target)**2
optimizer = torch.optim.Adam(model.parameters(), lr=0.2)

epochs = 200
for t in tqdm(range(epochs)):
    # print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, printing=True)
    # test(test_dataloader, model, loss_fn)
    # scheduler.step()
print("Done!")
print(train(train_dataloader, model, loss_fn, optimizer, printing=True))
print(test(test_dataloader, model, loss_fn))
# -


