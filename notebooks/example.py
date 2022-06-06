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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# We are going to recreate some figures for the "Small Synthetic Dataset" from [insert link to paper].

# +
import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader
from pennylane import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pennylane as pl

device = 'cpu' # Cuda support for PennyLane is possible though
# -

import sys
sys.path.insert(0, "../")
import utils.utils as utils
import models.fourier_models as fm
import models.quantum_models as qm

# +
# generate our regression task
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

n_samples = 200
n_features = 3
n_informative = 3
n_targets = 1
noise = 1.0
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


# +
def train(dataloader, model, loss_fn, optimizer):
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

# # The Quantum Model

n_qubits = n_features
model = qm.QuantumRegressionModel(n_qubits, n_layers=2, n_trainable_block_layers=1)
model.to(device)
loss_fn = nn.MSELoss(reduction='mean') 
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, line_search_fn="strong_wolfe")

# +
# Training
NN_loss = []
NN_test_loss = []

NN_loss.append(test(train_dataloader, model, loss_fn))
NN_test_loss.append(test(test_dataloader, model, loss_fn)) 
epochs = 50
for t in tqdm(range(epochs)):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    NN_loss.append(test(train_dataloader, model, loss_fn))
    NN_test_loss.append(test(test_dataloader, model, loss_fn))    
    print("Done!")
# -

x = np.arange(0, len(NN_loss), 1)
plt.plot(x, NN_loss, label="training loss")
plt.plot(x, NN_test_loss, label="test loss")
plt.xlabel("Epochs")
plt.ylabel("MSE loss")
plt.legend()

# # The Classical Surrogate

# +
# generate frequencies
max_freq = 2
dim = X_scaled[0].shape[0]

W = utils.freq_generator(max_freq, dim)

# we can solve the problem directly as it is a linear least squares problem, 
# giving the bestapproximation w.r.t. the training data
ba_coeffs = utils.fourier_best_approx(W, X_train, y_train) # vector c in paper

ba_loss = utils.loss(W, ba_coeffs, X_train, y_train).item()
ba_test_loss = utils.loss(W, ba_coeffs, X_test, y_test).item()

# +
# more in the fashion of ML, we can also train on the data with e.g. LBFGS
NN_loss = []
NN_test_loss = []
W.to(device)
model = fm.Fourier_model(W)
model.to(device)
loss_fn = nn.MSELoss(reduction='mean') # equiv. to torch.linalg.norm(input-target)**2
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, history_size=50, line_search_fn="strong_wolfe")

epochs = 50
for t in tqdm(range(epochs)):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    NN_loss.append(test(train_dataloader, model, loss_fn))
    NN_test_loss.append(test(test_dataloader, model, loss_fn))    
    print("Done!")
# -

x = np.arange(0, len(NN_loss), 1)
plt.plot(x, NN_loss, color="C0", label="training loss")
plt.plot(x, NN_test_loss, color="C1", label="test loss")
plt.plot(x, np.ones_like(x)*ba_loss, linestyle="dashed", color="C0", label="ba training loss")
plt.plot(x, np.ones_like(x)*ba_test_loss, linestyle="dashed", color="C1", label="ba test loss")
plt.xlabel("Epochs")
plt.ylabel("MSE loss")
plt.legend()


