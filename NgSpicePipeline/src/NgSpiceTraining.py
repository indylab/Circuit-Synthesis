from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from Training import models, utils, dataset
import trainingUtils
from torch import optim
import torch
import numpy as np


def check_acc(y_hat, y, margin=0.05):
    a_err = (np.abs(y_hat - y))  # get normalized error
    err = np.divide(a_err, y, out=a_err, where=y != 0)
    assert (err.shape == y.shape)
    num_correct = 0
    for row in err:
        num_in_row = len(np.where(row < margin)[0])  # margin * 100 because
        if num_in_row == len(row):
            num_correct += 1

    num_samples = y.shape[0]
    # correct_idx = np.where(err < margin)
    # num_part_correct = len(correct_idx[0])
    # num_part_samples = y.shape[0] * y.shape[1]
    print(f"Correct = {num_correct} / {num_samples}")
    return num_correct / num_samples


def train(model, train_data, val_data, optimizer, loss_fn, num_epochs=1000):
    print_every = 50
    accs = []
    losses = []
    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0
        for t, (x, y) in enumerate(train_data):
            # Zero your gradient
            optimizer.zero_grad()
            x_var = torch.autograd.Variable(x.type(torch.FloatTensor))
            y_var = torch.autograd.Variable(y.type(torch.FloatTensor).float())

            scores = model(x_var)

            loss = loss_fn(scores.float(), y_var.float())

            loss = torch.clamp(loss, max=500000, min=-500000)
            avg_loss += (loss.item() - avg_loss) / (t + 1)
            loss.backward()
            optimizer.step()

            losses.append(avg_loss)
            if (epoch + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (epoch + 1, avg_loss))

        if epoch % 50 == 0:
            for t, (x, y) in enumerate(val_data):
                paramater_preds = model(x)
                x1, x2 = paramater_preds[:, 0], paramater_preds[:, 1]
                x_sim, y_sim = trainingUtils.runSimulation(x1, x2)
                accs.append(check_acc(y_sim, y))
                print(f"Accuracy at Epoch {epoch} = {accs[-1]}")
    return losses, accs


if __name__ == '__main__':
    model = models.Model500GELU(2, 3)

    param_outfile_names = ["r.csv", "w.csv"]  # must be in order
    perform_outfile_names = ["bw.csv", "pw.csv", "a0.csv"]  # must be in order
    out = "C:\\Users\\tjtom\\OneDrive\\Desktop\\File_Cabinet\\Code\\NgSpicePipeline\\out\\"
    x, y = trainingUtils.getData(param_outfile_names, perform_outfile_names, out)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.L1Loss()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.9)
    train_data = dataset.CircuitSynthesisGainAndBandwidthManually(X_train, y_train)
    val_data = dataset.CircuitSynthesisGainAndBandwidthManually(X_test, y_test)
    train_dataloader = DataLoader(train_data, batch_size=100)
    val_dataloader = DataLoader(val_data, batch_size=100)

    losses, accs = train(model, train_dataloader, val_dataloader, optimizer, loss_fn)
