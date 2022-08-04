import os

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from Training import models, dataset
import trainingUtils
from torch import optim, cuda
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Simulator import Simulator
from visualutils import *


def check_acc(y_hat, y, margins=None):
    if margins is None:
        margins = [0.01, 0.05, 0.1]
    a_err = (np.abs(y_hat - y))  # get normalized error
    err = np.divide(a_err, y, where=y != 0)
    assert (err.shape == y.shape)

    accs = []
    for m in margins:
        num_correct = 0
        for row in err:
            num_in_row = len(np.where(row < m)[0])  # margin * 100 because
            if num_in_row == len(row):
                num_correct += 1
        num_samples = y.shape[0]
        print(f"{m}% num correct = {num_correct} / {num_samples}")
        accs.append(num_correct / num_samples)

    return accs


def check_minimum_requirement_acc(y_hat, y):

    greater = y_hat >= y

    return [np.all(greater, axis=1).sum().item() / y_hat.shape[0]]

def simulate_points(paramater_preds, norm_perform, scaler, simulator, margin):
    paramater_len = paramater_preds.shape[1]
    data = np.hstack((paramater_preds, norm_perform))
    MAX_LENGTH = 750
    if data.shape[0] > MAX_LENGTH:
        n = np.random.randint(0, paramater_preds.shape[0], MAX_LENGTH)
        data = data[n, :]
    unnorm_param_preds, unnorm_true_perform = scaler.inverse_transform(data)[:, :paramater_len], scaler.inverse_transform(
        data)[:, paramater_len:]

    _, y_sim = simulator.runSimulation(unnorm_param_preds)
    assert y_sim.shape == norm_perform.shape or y_sim.shape[
        0] == MAX_LENGTH, f"simulation failed, {y_sim.shape} != {norm_perform.shape}"
    if margin:
        accs = check_acc(y_sim, unnorm_true_perform)
    else:
        accs = check_minimum_requirement_acc(y_sim, unnorm_true_perform)
    return accs


def train(model, train_data, val_data, optimizer, loss_fn, scaler, simulator, device='cpu', num_epochs=1000, margin=True, train_acc = False):
    print_every = 50
    train_accs = []
    val_accs = []
    losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0
        val_avg_loss = 0
        for t, (x, y) in enumerate(train_data):
            # Zero your gradient
            optimizer.zero_grad()
            x_var = torch.autograd.Variable(x.type(torch.FloatTensor)).to(device)
            y_var = torch.autograd.Variable(y.type(torch.FloatTensor).float()).to(device)

            scores = model(x_var)

            loss = loss_fn(scores.float(), y_var.float())

            loss = torch.clamp(loss, max=500000, min=-500000)
            avg_loss += (loss.item() - avg_loss) / (t + 1)
            loss.backward()
            optimizer.step()

        losses.append(avg_loss)
        for t, (x, y) in enumerate(val_data):
            # Zero your gradient

            x_var = torch.autograd.Variable(x.type(torch.FloatTensor)).to(device)
            y_var = torch.autograd.Variable(y.type(torch.FloatTensor).float()).to(device)
            model.eval()
            scores = model(x_var)

            loss = loss_fn(scores.float(), y_var.float())

            loss = torch.clamp(loss, max=500000, min=-500000)
            val_avg_loss += (loss.item() - val_avg_loss) / (t + 1)

        losses.append(avg_loss)
        val_losses.append(val_avg_loss)
        if (epoch + 1) % print_every == 0:
            print('t = %d, loss = %.4f' % (epoch + 1, avg_loss))
            print('t = %d, val loss = %.4f' % (epoch + 1, val_avg_loss))

        if epoch % 50 == 0:
            norm_perform, _ = val_data.dataset.getAll()
            model.eval()
            paramater_preds = model(torch.Tensor(norm_perform)).detach().numpy()
            acc_list = simulate_points(paramater_preds, norm_perform, scaler, simulator, margin)
            val_accs.append(acc_list)
            print(f"Validation Accuracy at Epoch {epoch} = {val_accs[-1][0]}")
            if train_acc:
                norm_perform, _ = train_data.dataset.getAll()
                model.eval()
                paramater_preds = model(torch.Tensor(norm_perform)).detach().numpy()
                acc_list = simulate_points(paramater_preds, norm_perform, scaler, simulator, margin)
                train_accs.append(acc_list)
                print(f"Training_Accuracy at Epoch {epoch} = {train_accs[-1][0]}")

    return losses, val_losses, train_accs, val_accs



def get_subsetdata_accuracy(X_train, y_train, X_test, y_test, percentages, optims, loss_fn, scaler_arg, simulator, device = 'cpu'):
    #TODO different margin accuracy
    accuracy_list = []

    for percentage in percentages:
        model = models.Model50GELU(3, 2).to(device)
        optimizer = optims(model.parameters(), lr=0.001)
        subset_index = np.random.choice(np.arange(X_train.shape[0]), int(percentage * X_train.shape[0]), replace=False)
        new_X_train = X_train[subset_index, :]
        new_Y_train = y_train[subset_index, :]
        train_data = dataset.CircuitSynthesisGainAndBandwidthManually(new_X_train, new_Y_train)
        val_data = dataset.CircuitSynthesisGainAndBandwidthManually(X_test, y_test)
        train_dataloader = DataLoader(train_data, batch_size=100)
        val_dataloader = DataLoader(val_data, batch_size=100)
        _, _, _, val_accs = train(model, train_dataloader, val_dataloader, optimizer, loss_fn, scaler_arg,
                                             simulator, device, num_epochs=300)

        accs = []
        for acc in val_accs:
            accs.append(acc[-2])

        accuracy_list.append(accs)

    for index, acc in enumerate(accuracy_list):
        plt.plot(range(len(acc)), acc, label=percentages[index])
    plt.legend()
    plt.show()



