import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from Training import models, dataset
import trainingUtils
from torch import optim, cuda
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def check_acc(y_hat, y, margins=None):
    if margins is None:
        margins = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
    a_err = (np.abs(y_hat - y))  # get normalized error
    err = np.divide(a_err, y, out=a_err, where=y != 0)
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

    # correct_idx = np.where(err < margin)
    # num_part_correct = len(correct_idx[0])
    # num_part_samples = y.shape[0] * y.shape[1]

    return accs


def simulate_points(paramater_preds, norm_perform, scaler):
    data = np.hstack((paramater_preds, norm_perform))
    MAX_LENGTH = 750
    if data.shape[0] > MAX_LENGTH:
        n = np.random.randint(0,paramater_preds.shape[0],MAX_LENGTH)
        data = data[n,:]
    unnorm_param_preds, unnorm_true_perform = scaler.inverse_transform(data)[:, :2], scaler.inverse_transform(
        data)[:, 2:]
    param1, param2 = unnorm_param_preds[:, 0], unnorm_param_preds[:, 1]

    _, y_sim = trainingUtils.runSimulation(param1, param2)
    assert y_sim.shape == norm_perform.shape or y_sim.shape[0] == MAX_LENGTH, f"simulation failed, {y_sim.shape} != {norm_perform.shape}"
    accs = check_acc(y_sim, unnorm_true_perform)
    return accs


def train(model, train_data, val_data, optimizer, loss_fn, scaler, num_epochs=1000):
    print_every = 50
    train_accs = []
    val_accs = []
    losses = []
    train_acc = True
    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0
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
        if (epoch + 1) % print_every == 0:
            print('t = %d, loss = %.4f' % (epoch + 1, avg_loss))

        if epoch % 50 == 0:
            norm_perform, _ = val_data.dataset.getAll()
            model.eval()
            paramater_preds = model(torch.Tensor(norm_perform)).detach().numpy()
            acc_list = simulate_points(paramater_preds, norm_perform, scaler)
            val_accs.append(acc_list)
            print(f"Validation Accuracy at Epoch {epoch} = {val_accs[-1][0]}")
            if train_acc:
                norm_perform, _ = train_data.dataset.getAll()
                model.eval()
                paramater_preds = model(torch.Tensor(norm_perform)).detach().numpy()
                acc_list = simulate_points(paramater_preds, norm_perform, scaler)
                train_accs.append(acc_list)
                print(f"Training_Accuracy at Epoch {epoch} = {train_accs[-1][0]}")

    return losses, train_accs,val_accs


if __name__ == '__main__':

    # TODO: Change Run simulation and run_training methods to be a "simulator" class defend by netlist,
    #  netlist arguments and ngspice executable

    device = 'cuda:0' if cuda.is_available() else 'cpu'
    print(device)
    model = models.Model50GELU(3, 2).to(device)
    rerun_training = False
    if rerun_training:
        x, y = trainingUtils.run_training()
    else:
        param_outfile_names = ["r.csv", "w.csv"]  # must be in order
        perform_outfile_names = ["bw.csv", "pw.csv", "a0.csv"]  # must be in order
        out = r"NgSpicePipeline/out/"
        x, y = trainingUtils.getData(param_outfile_names, perform_outfile_names, out)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.L1Loss()

    data = np.hstack((x, y))
    scaler_arg = MinMaxScaler()
    scaler_arg.fit(data)
    data = scaler_arg.transform(data)
    param, perform = data[:, :2], data[:, 2:]
    print(param.shape, perform.shape)
    X_train, X_test, y_train, y_test = train_test_split(perform, param, test_size=0.1)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    train_data = dataset.CircuitSynthesisGainAndBandwidthManually(X_train, y_train)
    val_data = dataset.CircuitSynthesisGainAndBandwidthManually(X_test, y_test)
    train_dataloader = DataLoader(train_data, batch_size=100)
    val_dataloader = DataLoader(val_data, batch_size=100)

    losses, accs = train(model, train_dataloader, val_dataloader, optimizer, loss_fn, scaler_arg, num_epochs=1000)

    plt.plot(range(len(losses)), losses)
    plt.show()
    plt.plot(range(len(accs)), accs)
    plt.show()
