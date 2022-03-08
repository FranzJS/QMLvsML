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

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

import sys
sys.path.insert(0, "../")
import utils.utils as utils
import models.fourier_models as fm

# +
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X = torch.from_numpy(data.data)
y = torch.from_numpy(data.target)

X_scaled = utils.data_scaler(X, interval=(-torch.pi/2, torch.pi/2))

# Split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# +
# generate frequencies
max_freq = 2
dim = X_scaled[0].shape[0]

W = utils.freq_generator(max_freq, dim)
# -

# compute best approximation
ba_coeffs = utils.fourier_best_approx(W, X_train, y_train)

print("training_loss: ",utils.loss(W, ba_coeffs, X_train, y_train))
print("test loss: ",utils.loss(W, ba_coeffs, X_test, y_test))

ba_loss = []
ba_test_loss = []
for max_freq in range(1, 10+1):
    W = utils.freq_generator(max_freq, dim)
    ba_coeffs = utils.fourier_best_approx(W, X_train, y_train)
    ba_loss.append(utils.loss(W, ba_coeffs, X_train, y_train).item()**2)
    ba_test_loss.append(utils.loss(W, ba_coeffs, X_test, y_test).item()**2)
    print("max_freq: ", max_freq)
    print("training_loss: ",utils.loss(W, ba_coeffs, X_train, y_train).item()**2)
    print("test loss: ",utils.loss(W, ba_coeffs, X_test, y_test).item()**2)
    print("parameters: ", len(ba_coeffs))


# +
# -----------------------------------------

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
train_data_list = []
for i in tqdm(range(len(X_train))):
    data_point = (X_train[i], y_train[i])
    train_data_list.append(data_point)

test_data_list = []
for i in tqdm(range(len(X_test))):
    data_point = (X_test[i], y_test[i])
    test_data_list.append(data_point)
# -

train_dataloader = DataLoader(train_data_list, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data_list, batch_size=64, shuffle=True)

dim = X_scaled[0].shape[0]
W = utils.freq_generator(4, dim)
model = fm.Fourier_model(W)
model.to(device)
loss_fn = nn.MSELoss(reduction='sum') # equiv. to torch.linalg.norm(input-target)**2
optimizer = torch.optim.Adam(model.parameters())
# scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

NN_loss = []
NN_test_loss = []
for max_freq in range(1, 5+1):
    W = utils.freq_generator(max_freq, dim).to(device)
    model = fm.Fourier_model(W)
    model.to(device)
    loss_fn = nn.MSELoss(reduction='sum') # equiv. to torch.linalg.norm(input-target)**2
    optimizer = torch.optim.Adam(model.parameters(), lr=5*1e-3)

    epochs = 1000
    for t in tqdm(range(epochs)):
        # print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        # test(test_dataloader, model, loss_fn)
        # scheduler.step()
    print("Done!")
    NN_loss.append(train(train_dataloader, model, loss_fn, optimizer, printing=True))
    NN_test_loss.append(test(test_dataloader, model, loss_fn))

w = np.arange(1, 10+1, 1)
#plt.plot(w, ba_loss)
plt.plot(w, NN_loss)

save = True
save_path = "../data/california_01/"
if save == True:
    np.save(save_path+"NN_loss.npy", np.array(NN_loss))
    np.save(save_path+"NN_test_loss.npy", np.array(NN_test_loss))
    np.save(save_path+"ba_loss.npy", np.array(ba_loss))
    np.save(save_path+"ba_test_loss.npy", np.array(ba_test_loss))

plt.plot(w, ba_test_loss)
plt.plot(w, NN_test_loss)
plt.ylim(0, 100)

NN_loss

10**8

