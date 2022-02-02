import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import utils
import dataloader


class CSGainAndBandwidthManually(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(CSGainAndBandwidthManually, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 30),
            nn.ReLU(inplace=True),
            nn.Linear(30, 30),
            nn.ReLU(inplace=True),
            nn.Linear(30, output_count)
        )

    def forward(self, x):
        return self.network(x)


def train(model, training_data, loss_fn, optimizer, dtype=torch.FloatTensor, num_epochs=1, print_every=10):
    loss_list = []
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(training_data):

            x_var = torch.autograd.Variable(x.type(dtype))
            y_var = torch.autograd.Variable(y.type(dtype).float())

            # make predictions
            scores = model(x_var)

            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.item()))

            # Zero your gradient
            optimizer.zero_grad()
            # Compute the loss gradients
            loss.backward()
            # Adjust learning weights
            optimizer.step()
        loss_list.append(loss.item())
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
    for x, y in loader:
        with torch.no_grad():
            x_var = torch.autograd.Variable(x.type(dtype))

        y_hat = model(x_var)

        y_hat = np.array(y_hat.detach(), dtype)
        y = np.array(y.detach(), dtype)

        # TODO add functionality to make this a parameter so we can try multiple accuracy measures
        err = np.mean(np.square(y_hat - y), axis=1)  # mse formula

        num_correct += len(np.where(err < margin)[0])
        num_samples += y_hat.shape[0]
        all_preds.extend(np.array(y_hat))
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc, all_preds


def mockSimulator(xy):
    np.random.seed(123)
    input = xy

    A = np.random.rand(1, 5)
    B = np.random.rand(5, 2)
    C = np.random.rand(2, 15)

    iA = input.dot(A)
    iAB = iA.dot(B)
    full = iAB.dot(C)

    ret = np.array([np.mean(full[0]), np.mean(full[1])]).reshape(2, 1)

    return ret


if __name__ == '__main__':
    # TODO turn this script into a notebook for better visualization
    test = torch.rand(10, 2)
    test_model = CSGainAndBandwidthManually()

    test_out = test_model(test)

    # # test mock simulator
    # print(mockSimulator(np.array([9,11]).reshape(2,1)))
    # print(mockSimulator(np.array([1,1]).reshape(2,1)))
    #
    # # create mini data set
    # sim_inputs = np.random.uniform(-100,100,(10000,2,1))
    # sim_outputs = np.zeros(sim_inputs.shape)
    # for i in range(len(sim_inputs)):
    #     sim_outputs[i,:] = mockSimulator(sim_inputs[i])

    # load datasets and split into train and val sets
    # TODO implement randomness in dataset splitting
    data = np.array(utils.parseGainAndBWCsv('../Data/BW_Gain.csv')).astype(float)
    train_dataset = dataloader.CircuitSynthesisGainAndBandwidthManually(data[:120, 1], data[:120, 0])
    val_dataset = dataloader.CircuitSynthesisGainAndBandwidthManually(data[120:, 1], data[120:, 0])

    dtype = torch.FloatTensor
    loss_fn = nn.L1Loss().type(dtype)  # loss can be changed here. This is the first one i tried that gave meaningful results
    optimizer = optim.SGD(test_model.parameters(), lr=.00001)  # TODO haven't experimented with this yet

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
