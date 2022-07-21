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


def simulate_points(paramater_preds, norm_perform, scaler, simulator):
    data = np.hstack((paramater_preds, norm_perform))
    MAX_LENGTH = 750
    if data.shape[0] > MAX_LENGTH:
        n = np.random.randint(0, paramater_preds.shape[0], MAX_LENGTH)
        data = data[n, :]
    unnorm_param_preds, unnorm_true_perform = scaler.inverse_transform(data)[:, :3], scaler.inverse_transform(
        data)[:, 3:]

    _, y_sim = simulator.runSimulation(unnorm_param_preds)
    assert y_sim.shape == norm_perform.shape or y_sim.shape[
        0] == MAX_LENGTH, f"simulation failed, {y_sim.shape} != {norm_perform.shape}"
    accs = check_acc(y_sim, unnorm_true_perform)
    return accs


def train(model, train_data, val_data, optimizer, loss_fn, scaler, simulator, num_epochs=1000):
    print_every = 50
    train_accs = []
    val_accs = []
    losses = []
    val_losses = []
    train_acc = False
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
            acc_list = simulate_points(paramater_preds, norm_perform, scaler, simulator)
            val_accs.append(acc_list)
            print(f"Validation Accuracy at Epoch {epoch} = {val_accs[-1][0]}")
            if train_acc:
                norm_perform, _ = train_data.dataset.getAll()
                model.eval()
                paramater_preds = model(torch.Tensor(norm_perform)).detach().numpy()
                acc_list = simulate_points(paramater_preds, norm_perform, scaler, simulator)
                train_accs.append(acc_list)
                print(f"Training_Accuracy at Epoch {epoch} = {train_accs[-1][0]}")

    return losses, val_losses, train_accs, val_accs


if __name__ == '__main__':

    # TODO: Change Run simulation and run_training methods to be a "simulator" class defend by netlist,
    #  netlist arguments and ngspice executable

    device = 'cuda:0' if cuda.is_available() else 'cpu'
    print(device)

    ngspice_exec = "ngspice/Spice64/bin/ngspice.exe"
    train_netlist_nmos = "NgSpicePipeline/assets/nmos-training.sp"
    test_netlist_nmos = "NgSpicePipeline/assets/nmos-testing-pro.sp"
    param_list_nmos = ["r", "w"]
    perform_list_nmos = ["bw", "pw", "a0"]

    arguments_nmos = {
        "model_path": "NgSpicePipeline/assets/45nm_CS.pm",
        "w_start": 620,
        "w_stop": 1450,
        "w_change": 11,
        "r_start": "2.88u",
        "r_stop": "6.63u",
        "r_change": "0.3750u",
        "out": "NgSpicePipeline/out/"
    }
    simulator_nmos = Simulator(ngspice_exec, train_netlist_nmos, test_netlist_nmos, param_list_nmos, perform_list_nmos,
                               arguments_nmos)

    train_netlist_cascade = "NgSpicePipeline/assets/nmos-training-cascode.sp"
    test_netlist_cascade = "NgSpicePipeline/assets/nmos-testing-cascode.sp"
    param_list_cascade = ["r", "w0", "w1"]
    perform_list_cascade = ["bw", "pw", "a0"]

    arguments_cascade = {
        "model_path": "NgSpicePipeline/assets/45nm_CS.pm",
        "w0_start": 620,
        "w0_stop": 1450,
        "w0_change": 50,
        "w1_start": 620,
        "w1_stop": 1450,
        "w1_change": 50,
        "r_start": "2.88u",
        "r_stop": "6.63u",
        "r_change": "0.7500u",
        "out": "NgSpicePipeline/out/"
    }
    simulator_cascade = Simulator(ngspice_exec, train_netlist_cascade, test_netlist_cascade, param_list_cascade,
                                  perform_list_cascade,
                                  arguments_cascade)

    train_netlist_two_stage = "NgSpicePipeline/assets/TwoStageAmplifier.sp"
    test_netlist_two_stage = "NgSpicePipeline/assets/TwoStageAmplifier-Test.sp"
    param_list_two_stage = ["w0", "w1", "w2"]
    perform_list_two_stage = ["bw", "pw", "a0"]

    arguments_two_stage = {
        "model_path": "NgSpicePipeline/assets/45nm_CS.pm",
        "w0_start": "25u",
        "w0_stop": "30u",
        "w0_change": "0.5u",
        "w2_start": "52u",
        "w2_stop": "55.5u",
        "w2_change": "0.5u",
        "w1_start": "6u",
        "w1_stop": "9u",
        "w1_change": "0.5u",
        "out": "NgSpicePipeline/out/"
    }
    simulator_two_stage = Simulator(ngspice_exec, train_netlist_two_stage, test_netlist_two_stage, param_list_two_stage,
                                    perform_list_two_stage,
                                    arguments_two_stage)
    simulator_two_stage.delete_existing_data = True

    simulator = simulator_two_stage
    rerun_training = True
    if rerun_training:
        x, y = simulator.run_training()
    else:
        param_outfile_names = ["w0.csv", "w1.csv", "w2.csv"]  # must be in order
        perform_outfile_names = ["bw.csv", "pw.csv", "a0.csv"]  # must be in order
        out = r"NgSpicePipeline/out/"
        x, y = simulator.getData(param_outfile_names, perform_outfile_names, out)

    model = models.Model50GELU(3, 3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.L1Loss()
    print(x.shape, y.shape)
    data = np.hstack((x, y))
    scaler_arg = MinMaxScaler()
    scaler_arg.fit(data)
    data = scaler_arg.transform(data)
    param, perform = data[:, :3], data[:, 3:]
    print(param.shape, perform.shape)
    X_train, X_test, y_train, y_test = train_test_split(perform, param, test_size=0.1)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    train_data = dataset.CircuitSynthesisGainAndBandwidthManually(X_train, y_train)
    val_data = dataset.CircuitSynthesisGainAndBandwidthManually(X_test, y_test)
    train_dataloader = DataLoader(train_data, batch_size=100)
    val_dataloader = DataLoader(val_data, batch_size=100)

    losses, val_losses, train_accs, val_accs = train(model, train_dataloader, val_dataloader, optimizer, loss_fn,
                                                     scaler_arg,
                                                     simulator, num_epochs=1000)

    plt.plot(range(len(losses)), losses)
    plt.xlabel("epochs")
    plt.ylabel("train loss")
    plt.show()
    plt.plot(range(len(val_losses)), val_losses)
    plt.xlabel("epochs")
    plt.ylabel("validation loss")
    plt.show()
    plt.plot(range(len(val_accs)), val_accs)
    plt.xlabel("epochs")
    plt.ylabel("validation accuracy")
    plt.legend(['1%', '5%', '10%'])
    plt.show()
