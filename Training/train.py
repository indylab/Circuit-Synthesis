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


def train(model, training_data, loss_fn, optimizer, dtype=torch.FloatTensor, num_epochs=1, print_every=10):
    #Regular Training Loop
    loss_list = []
    for epoch in range(num_epochs):
        #print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        avg_loss = 0
        for t, (x, y) in enumerate(training_data):
            x_var = torch.autograd.Variable(x.type(dtype).float())
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
        if (epoch + 1) % print_every == 0:
            print('t = %d, loss = %.4f' % (epoch + 1, avg_loss))
        loss_list.append(avg_loss)
    return loss_list

def trainProbModel(model, training_data, loss_fn, optimizer, dtype=torch.FloatTensor, num_epochs=1, print_every=10):
    #Regular Training Loop
    loss_list = []
    for epoch in range(num_epochs):
        #print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        avg_loss = 0
        for t, (x, y) in enumerate(training_data):
            x_var = torch.autograd.Variable(x.type(dtype).float())
            y_var = torch.autograd.Variable(y.type(dtype).float())

            # make predictions
            scores = model(x_var)
            inputs = scores[:,:2]
            var = scores[:,2:]
            loss = loss_fn(inputs.float(), y_var.float(), var.float())
            #TODO implement proper loss or gradient clipping
            loss = torch.clamp(loss, max = 500000, min = -500000)
            avg_loss += (loss.item() - avg_loss) / (t+1)


            # Zero your gradient
            optimizer.zero_grad()
            # Compute the loss gradients
            loss.backward()
            # Adjust learning weights
            optimizer.step()
        if (epoch + 1) % print_every == 0:
            print('t = %d, loss = %.4f' % (epoch + 1, avg_loss))
        loss_list.append(avg_loss)
    return loss_list
def trainWValidation(model, training_data, val_data, loss_fn, optimizer, dtype=torch.FloatTensor, num_epochs=1, print_every=10, wTune = False, margin = 0.05):
    """
    Training Loop with Validation check at each step.
    Params:
        wTune (Bool): if your using raytune wTune = True reports accuracy and part accuracy and loss per epoch
        margin (float): is the percent error your willing to allow.
    """
    loss_list = []
    acc_list = []
    part_acc_list = []
    for epoch in range(num_epochs):
        #print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        avg_loss = 0
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
        if (epoch + 1) % print_every == 0:
            acc, part_acc, idx = check_accuracy(model,training_data,margin)
        else:
            acc, part_acc, idx = check_accuracy(model,training_data,margin, verbose=False)
        print('t = %d, loss = %.4f' % (epoch + 1, avg_loss))
        if wTune:
            tune.report(accuracy = acc, partial_accuracy = part_acc, validation_accuracy = val_acc, validation_partial_accuracy=val_part_acc, loss = avg_loss)
        loss_list.append(avg_loss)
        acc_list.append(acc)
        part_acc_list.append(part_acc)
    return loss_list, acc_list, part_acc_list

def check_accuracy(model, loader, margin, dtype=torch.FloatTensor, train=True, verbose = True):
    num_part_correct = 0
    num_part_samples = 0
    num_correct = 0
    num_samples = 0
    
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    loss_list = []
    all_preds = []
    for x, y in loader:
        with torch.no_grad():
            x_var = torch.autograd.Variable(x.type(dtype))

        y_hat = model(x_var)

        y_hat = np.array(y_hat.detach(), dtype)
        y = np.array(y.detach(), dtype)
        
       
        err = np.abs(y_hat - y)
        for row in err:
            num_in_row = len(np.where(row < margin)[0])
            if num_in_row == len(row):
                num_correct += 1

        num_samples += y.shape[0]
        correct_idx = np.where(err < margin)
        num_part_correct += len(correct_idx[0])
        num_part_samples += y.shape[0] * y.shape[1]
        
        all_preds.extend(y_hat)
    part_acc = float(num_part_correct) / num_part_samples
    acc = float(num_correct) / num_samples
    if verbose:
        print('Got %d / %d partially correct (%.2f pct)' % (num_part_correct, num_part_samples, 100 * part_acc)) 
        print('Got %d / %d correct (%.2f pct)' % (num_correct, num_samples, 100 * acc)) 
    return acc, part_acc, all_preds

def check_raw_accuracy(model, loader, min_max, margin, dtype=torch.FloatTensor, verbose = True):
    num_part_correct = 0
    num_part_samples = 0
    num_correct = 0
    num_samples = 0
    
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    loss_list = []
    all_preds = []
    for x, y in loader:
        with torch.no_grad():
            x_var = torch.autograd.Variable(x.type(dtype))

        y_hat = model(x_var)

        y_hat = np.array(y_hat.detach(), dtype)
        y = np.array(y.detach(), dtype)
        
        y_min_max = min_max[2:,:] # discard x min_maxs
        y = denormalize(y,y_min_max)
        y_hat = denormalize(y_hat,y_min_max)
       
        err = np.abs(y_hat - y)/y
        for row in err:
            num_in_row = len(np.where(row < margin)[0])
            if num_in_row == len(row):
                num_correct += 1

        num_samples += y.shape[0]
        correct_idx = np.where(err < margin)
        num_part_correct += len(correct_idx[0])
        num_part_samples += y.shape[0] * y.shape[1]
        
        all_preds.extend(y_hat)
    part_acc = float(num_part_correct) / num_part_samples
    acc = float(num_correct) / num_samples
    if verbose:
        print('Got %d / %d partially correct (%.2f pct)' % (num_part_correct, num_part_samples, 100 * part_acc)) 
        print('Got %d / %d correct (%.2f pct)' % (num_correct, num_samples, 100 * acc)) 
    return acc, part_acc, all_preds

# two experimental custom loss functions. VERY basic
def L1MarginalLoss2(yhat, y, margin = 0.10):
    errs = (torch.abs(yhat - y) / y).type(torch.float64)
    errs = torch.where(errs > margin, errs, float(0.0))
    return torch.mean(errs,dtype=torch.float)

def L1MarginalLoss1(yhat, y, margin = 0.10):
    errs = (torch.abs(yhat - y)).type(torch.float64)
    errs = torch.where(errs > margin, errs, float(0.0))
    return torch.mean(errs,dtype=torch.float)

def check_guassian_raw_accuracy(model, loader, min_max, margin, dtype=torch.FloatTensor, verbose = True):
    num_part_correct = 0
    num_part_samples = 0
    num_correct = 0
    num_samples = 0
    
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    loss_list = []
    all_preds = []
    for x, y in loader:
        with torch.no_grad():
            x_var = torch.autograd.Variable(x.type(dtype))

        y_hat = model(x_var)
        
        y_hat = np.array(y_hat.detach())
        var1_expectation = y_hat[:,0]
        var2_expectation = y_hat[:,1]
        var1_std = np.exp(y_hat[:,2])
        var2_std = np.exp(y_hat[:,3])
        
        var1 = np.random.normal(var1_expectation, var1_std)[:,None]
        var2 = np.random.normal(var2_expectation, var2_std)[:,None]
        y_hat = np.concatenate((var1,var2), axis = 1)

        y_min_max = min_max[2:] # discard x min_maxs
        y = denormalize(y,y_min_max)
        y_hat = denormalize(y_hat,y_min_max)
       
        err = np.abs(y_hat - y)/y
        for row in err:
            num_in_row = len(np.where(row < margin)[0])
            if num_in_row == len(row):
                num_correct += 1

        num_samples += y.shape[0]
        correct_idx = np.where(err < margin)
        num_part_correct += len(correct_idx[0])
        num_part_samples += y.shape[0] * y.shape[1]
        
        all_preds.extend(y_hat)
    part_acc = float(num_part_correct) / num_part_samples
    acc = float(num_correct) / num_samples
    if verbose:
        print('Got %d / %d partially correct (%.2f pct)' % (num_part_correct, num_part_samples, 100 * part_acc)) 
        print('Got %d / %d correct (%.2f pct)' % (num_correct, num_samples, 100 * acc)) 
    return acc, part_acc, all_preds

