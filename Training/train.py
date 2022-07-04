import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .utils import *
# from dataset import *
# from model import *
import numpy as np
from ray import tune


def trainProbModel(model, training_data, loss_fn, optimizer, dtype=torch.FloatTensor, num_epochs=1, print_every=10, validation_data = None):
    #Regular Training Loop
    loss_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        #print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        avg_loss = 0
        val_avg_loss = 0
        for t, (x, y) in enumerate(training_data):
            x_var = torch.autograd.Variable(x.type(dtype).float())
            y_var = torch.autograd.Variable(y.type(dtype).float())
            
             # Zero your gradient
            optimizer.zero_grad()
            # make predictions
            scores = model(x_var)
            inputs = scores[:,:2]
            var = scores[:,2:]
            loss = loss_fn(inputs.float(), y_var.float(), var.float())
            #TODO implement proper loss or gradient clipping
            #loss = torch.clamp(loss, max = 500000, min = -500000)
            avg_loss += (loss.item() - avg_loss) / (t+1)


            # Zero your gradient
            #optimizer.zero_grad()
            # Compute the loss gradients
            loss.backward()
            # Adjust learning weights
            optimizer.step()
        if validation_data:
            model.eval()
            for t, (xv, yv) in enumerate(validation_data):
                xv_var = torch.autograd.Variable(xv.type(dtype).float())
                yv_var = torch.autograd.Variable(yv.type(dtype).float())

                # make predictions
                scores_val = model(xv_var)
                inputs_val = scores_val[:,:2]
                var_val = scores_val[:,2:]
                val_loss = loss_fn(inputs_val.float(), yv.float(), var_val.float())

                val_loss = torch.clamp(val_loss, max = 500000, min = -500000)
                val_avg_loss += (loss.item() - val_avg_loss) / (t+1)
            
        if (epoch + 1) % print_every == 0:
            print('t = %d, loss = %.4f, val loss = %.4f' % (epoch + 1, avg_loss, val_avg_loss))
        loss_list.append(avg_loss)
        val_loss_list.append(val_avg_loss)
    return loss_list, val_loss_list

def trainAbsoluteModel(model, training_data, loss_fn, optimizer, dtype=torch.FloatTensor, num_epochs=1, print_every=10, validation_data = None):
    """
    Training Loop with Validation check at each step.
    Params:
        wTune (Bool): if your using raytune wTune = True reports accuracy and part accuracy and loss per epoch
        margin (float): is the percent error your willing to allow.
    """
    loss_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        #print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        avg_loss = 0
        val_avg_loss = 0
        for t, (x, y) in enumerate(training_data):

            x_var = torch.autograd.Variable(x.type(dtype))
            y_var = torch.autograd.Variable(y.type(dtype).float())

            # make predictions
            scores = model(x_var)
            
            loss = loss_fn(scores.float(), y_var.float())

            #TODO implement proper loss or gradient clipping
            loss = torch.clamp(loss, max = 500000, min = -500000)
            avg_loss += (loss.item() - avg_loss) / (t+1)


            # Zero your gradient
            optimizer.zero_grad()
            # Compute the loss gradients
            loss.backward()
            # Adjust learning weights
            optimizer.step()
        if validation_data:
            model.eval()
            for t, (xv, yv) in enumerate(validation_data):
                xv_var = torch.autograd.Variable(xv.type(dtype).float())
                yv_var = torch.autograd.Variable(yv.type(dtype).float())

                # make predictions
                scores_val = model(xv_var)

                val_loss = loss_fn(scores_val.float(), yv.float())

                val_loss = torch.clamp(val_loss, max = 500000, min = -500000)
                val_avg_loss += (val_loss.item() - val_avg_loss) / (t+1)
        if (epoch + 1) % print_every == 0:
            print('t = %d, loss = %.4f, val loss = %.4f' % (epoch + 1, avg_loss, val_avg_loss))

        loss_list.append(avg_loss)
        val_loss_list.append(val_avg_loss)

    return loss_list, val_loss_list





# two experimental custom loss functions. VERY basic
def L1MarginalLoss2(yhat, y, margin = 0.10):
    errs = (torch.abs(yhat - y) / y).type(torch.float64)
    errs = torch.where(errs > margin, errs, float(0.0))
    return torch.mean(errs,dtype=torch.float)

def L1MarginalLoss1(yhat, y, margin = 0.10):
    errs = (torch.abs(yhat - y)).type(torch.float64)
    errs = torch.where(errs > margin, errs, float(0.0))
    return torch.mean(errs,dtype=torch.float)

