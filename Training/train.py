import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
from dataset import *
from model import *
import numpy as np

def train(model, training_data, loss_fn, optimizer, dtype=torch.FloatTensor, num_epochs=1, print_every=10):
    loss_list = []
    for epoch in range(num_epochs):
        #print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        avg_loss = 0
        for t, (x, y) in enumerate(training_data):

            x_var = torch.autograd.Variable(x.type(dtype))
            y_var = torch.autograd.Variable(y.type(dtype).float())

            # make predictions
            scores = model(x_var)

            loss = loss_fn(scores, y_var)
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


def check_accuracy(model, loader, margin, dtype=torch.FloatTensor, train=True):
    if train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    all_preds = []
    err_list = []
    for x, y in loader:
        with torch.no_grad():
            x_var = torch.autograd.Variable(x.type(dtype))

        y_hat = model(x_var)

        y_hat = np.array(y_hat.detach(), dtype)
        y = np.array(y.detach(), dtype)

        # TODO add functionality to make this a parameter so we can try multiple accuracy measures
        err = np.mean(np.square(y_hat - y), axis=1)  # mse formula

        err_list.append(np.mean(err))
        num_correct += len(np.where(err < margin)[0])
        num_samples += y_hat.shape[0]
        all_preds.extend(np.array(y_hat))
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    print('Error for model is {}'.format(round(sum(err_list) / len(err_list), 5)))
    return acc, all_preds


if __name__ == '__main__':
    # TODO turn this script into a notebook for better visualization

    test_model = CSGainAndBandwidthManually()


    # load datasets and split into train and val sets

    data = np.array(parseGainAndBWCsv('../Data/BW_Gain.csv')).astype(float)

    dataset = CircuitSynthesisGainAndBandwidthManually(data[:, 1], data[:, 0])
    train_dataset, val_dataset = splitDataset(dataset, 0.8)



    dtype = torch.FloatTensor
    loss_fn = nn.MSELoss().type(dtype)  # loss can be changed here. This is the first one i tried that gave meaningful results
    optimizer = optim.Adam(test_model.parameters(), lr=3e-4)  # TODO haven't experimented with this yet

    train_data = DataLoader(train_dataset, batch_size=5)
    validation_data = DataLoader(val_dataset, batch_size=5)

    # train nn and check accuracy
    losses = train(test_model, train_data, loss_fn, optimizer, num_epochs=2000, print_every=10)
    # TODO accuracy may not be right. this was a quick attempt
    acc, preds = check_accuracy(test_model, validation_data, .20)

    # plot losses per epoch
    plt.plot(losses)
    plt.show()

    # print(test_out.shape)
