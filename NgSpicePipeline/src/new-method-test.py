import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from NgSpiceTraining import *
from Training import models, dataset
from torch import optim, cuda
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Simulator import Simulator
from Training.dataset import CircuitSynthesisGainAndBandwidthManually
from visualutils import *
import os
from trainingUtils import *

if __name__ == '__main__':

    # TODO: Change Run simulation and run_training methods to be a "simulator" class defend by netlist,
    #  netlist arguments and ngspice executable

    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"
    # train_netlist_two_stage = "../assets/TwoStageAmplifier.sp"
    # test_netlist_two_stage = "../assets/TwoStageAmplifier-Test.sp"
    # param_list_two_stage = ["w0", "w1", "w2"]
    # perform_list_two_stage = ["bw", "pw", "a0"]
    #
    # arguments_two_stage = {
    #     "model_path": "../assets/45nm_CS.pm",
    #     "w0_start": "25u",
    #     "w0_stop": "30u",
    #     "w0_change": "0.25u",
    #     "w2_start": "52u",
    #     "w2_stop": "55.5u",
    #     "w2_change": "0.5u",
    #     "w1_start": "6u",
    #     "w1_stop": "9u",
    #     "w1_change": "0.5u",
    #     "out": "../out/"
    # }
    # simulator_two_stage = Simulator(ngspice_exec, train_netlist_two_stage, test_netlist_two_stage, param_list_two_stage,
    #                                 perform_list_two_stage,
    #                                 arguments_two_stage)
    train_netlist_nmos = "../assets/nmos-training.sp"
    test_netlist_nmos = "../assets/nmos-testing-pro.sp"
    param_list_nmos = ["r", "w"]
    perform_list_nmos = ["bw", "pw", "a0"]

    arguments_nmos = {
        "model_path": "../assets/45nm_CS.pm",
        "w_start": 620,
        "w_stop": 1450,
        "w_change": 5,
        "r_start": "2.88u",
        "r_stop": "6.63u",
        "r_change": "0.20u",
        "out": "../out/"
    }
    simulator_nmos = Simulator(ngspice_exec, train_netlist_nmos, test_netlist_nmos, param_list_nmos, perform_list_nmos,
                               arguments_nmos)

    train_netlist_cascade = "../assets/nmos-training-cascode.sp"
    test_netlist_cascade = "../assets/nmos-testing-cascode.sp"
    param_list_cascade = ["r", "w0", "w1"]
    perform_list_cascade = ["bw", "pw", "a0"]

    arguments_cascade = {
        "model_path": "../assets/45nm_CS.pm",
        "w0_start": 620,
        "w0_stop": 1450,
        "w0_change": 50,
        "w1_start": 620,
        "w1_stop": 1450,
        "w1_change": 50,
        "r_start": "2.88u",
        "r_stop": "6.63u",
        "r_change": "0.7500u",
        "out": "../out/"
    }
    simulator_cascade = Simulator(ngspice_exec, train_netlist_cascade, test_netlist_cascade, param_list_cascade,
                                  perform_list_cascade,
                                  arguments_cascade)

    train_netlist_two_stage = "../assets/TwoStageAmplifier.sp"
    test_netlist_two_stage = "../assets/TwoStageAmplifier-Test.sp"
    param_list_two_stage = ["w0", "w1", "w2"]
    perform_list_two_stage = ["bw", "pw", "a0"]

    arguments_two_stage = {
        "model_path": "../assets/45nm_CS.pm",
        "w0_start": "25u",
        "w0_stop": "30u",
        "w0_change": "0.5u",
        "w2_start": "52u",
        "w2_stop": "55.5u",
        "w2_change": "0.5u",
        "w1_start": "6u",
        "w1_stop": "9u",
        "w1_change": "0.5u",
        "out": "../out/"
    }
    simulator_two_stage = Simulator(ngspice_exec, train_netlist_two_stage, test_netlist_two_stage, param_list_two_stage,
                                    perform_list_two_stage,
                                    arguments_two_stage)
    simulator_two_stage.delete_existing_data = True

    train_netlist_lna = "../assets/LNA.sp"
    test_netlist_lna = "../assets/LNA_test"
    param_list_lna = ["ls", "ld", "lg", "r", "w"]
    perform_list_lna = ["Gmax", "Gp", "s11", "nf"]
    arguments_lna = {
        "model_path": "../assets/45nm_CS.pm",
        "ls_start": "58.3p",
        "ls_stop": "60.8p",
        "ls_change": "0.5p",
        "ld_start": "4.4n",
        "ld_stop": "5.4n",
        "ld_change": "0.5n",
        "lg_start": "14.8n",
        "lg_stop": "15.8n",
        "lg_change": "0.24n",
        "r_start": "800",
        "r_stop": "1050",
        "r_change": "50",
        "w_start": "51u",
        "w_stop": "53u",
        "w_change": "0.4u",
        "out": "../out/"
    }
    simulator_lna = Simulator(ngspice_exec, train_netlist_lna, test_netlist_lna, param_list_lna, perform_list_lna,
                              arguments_lna)
    simulator = simulator_cascade
    simulator.delete_existing_data = False

    rerun_training = False

    if rerun_training:
        x, y = simulator.run_training()
    else:
        param_outfile_names = simulator.train_param_filenames  # must be in order
        perform_outfile_names = simulator.train_perform_filenames  # must be in order
        curPath = os.getcwd()
        print(curPath)
        out = os.path.join(curPath, "../out/")

        x, y = simulator.getData(param_outfile_names, perform_outfile_names, out)

    # plot_parameter(x, y, reduce_dim_y=True)

    # device = 'cuda:0' if cuda.is_available() else 'cpu'
    device = 'cpu'
    print(device)
    num_param, num_perform = len(simulator.parameter_list), len(simulator.performance_list)
    model = models.Model50GELU(num_perform, num_param).to(device)

    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()
    print(x.shape, y.shape)
    data = np.hstack((x, y))
    scaler_arg = MinMaxScaler()
    scaler_arg.fit(data)
    data = scaler_arg.transform(data)
    param, perform = data[:, :num_param], data[:, num_param:]

    perform, param = generate_new_dataset_maximum_performance(perform, param, [0,2,1], [1, -1, 1])
    X_train, X_test, y_train, y_test = train_test_split(perform, param, test_size=0.1)

    # newX_train, newy_train = generate_duplicate_data(X_train, y_train, [0.98,0.96,0.94,0.92,0.9])

    train_dataset = CircuitSynthesisGainAndBandwidthManually(X_train, y_train)
    val_dataset = CircuitSynthesisGainAndBandwidthManually(X_test, y_test)

    train_data = DataLoader(train_dataset, batch_size=100)
    val_data = DataLoader(val_dataset, batch_size=100)
    losses, val_losses, train_accs, val_accs = train(model, train_data, val_data, optimizer, loss_fn, scaler_arg,
                                                     simulator, device='cpu', num_epochs=1000,
                                                     margin=False, sign=[1, -1, 1])
    # get_subsetdata_accuracy(X_train, y_train, X_test, y_test, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], optimizer, loss_fn, scaler_arg, simulator)

    plt.plot(range(len(losses)), losses)
    plt.show()
    plt.plot(range(len(val_accs)), val_accs)
    plt.show()
    plt.plot(range(len(train_accs)), train_accs)
    plt.show()
