import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from circuit import prepare_data
from eval_model import Model500GELU
from simulator import load_simulator
from utils import load_yaml, check_save_data_status, saveDictToTxt
from pipeline import generate_circuit_given_config
import os
from dataset import ArgMaxDataset, BasePytorchModelDataset


def useability_pipeline():
    useability_config_path = os.path.join(os.getcwd(), "config", "useability_config.yaml")
    pipeline_config = load_yaml(useability_config_path)
    circuit_config = generate_circuit_given_config(pipeline_config)
    simulator = load_simulator(circuit_config=circuit_config,
                                simulator_config=pipeline_config['simulator_config'])

    if not check_save_data_status(circuit_config):
        data_for_evaluation = prepare_data(simulator.parameter_list, simulator.arguments)

        start = time.time()
        print('start sim')
        parameter, performance = simulator.runSimulation(data_for_evaluation, True)
        print('took for sim', time.time() - start)
        print('Params shape', parameter.shape)
        print('Perfomance shape', performance.shape)

        print("Saving metadata for this simulation")
        metadata_path = os.path.join(circuit_config["arguments"]["out"], "metadata.txt")
        saveDictToTxt(circuit_config["arguments"], metadata_path)
    else:
        print("Load from saved data")
        parameter = np.load(os.path.join(simulator.arguments["out"], "x.npy"))
        performance = np.load(os.path.join(simulator.arguments["out"], "y.npy"))

    print(parameter.shape)
    print(performance.shape)
    dataset = ArgMaxDataset(circuit_config["order"], circuit_config["sign"], pipeline_config)
    norm_parameter, norm_performance, scaler = dataset.transform_data(parameter, performance)

    modify_parameter, modify_performance = dataset.modify_data(norm_parameter, norm_performance, None, None, True)

    model = Model500GELU(len(circuit_config["performance_list"]), len(circuit_config["parameter_list"]))

    torchDataset = BasePytorchModelDataset(modify_performance, modify_parameter)

    train_pytorch_model(model, torchDataset)

    performance_req_list = circuit_config["performance_list"]


    while True:
        performance_req = []
        for req in performance_req_list:
            req_value = input("Please Enter Performance requirements for field {}: \n".format(req))
            performance_req.append(float(req_value))
        print(performance_req)

        performance_req = np.array(performance_req)[None,:]
        random_parameter = np.random.rand(1, len(circuit_config["parameter_list"]))
        data = np.hstack((np.copy(random_parameter), np.copy(performance_req)))

        scaled_data = scaler.transform(data)

        norm_performance_req = scaled_data[:,len(circuit_config["parameter_list"]):]
        model.eval()
        predict_parameter = model(torch.Tensor(norm_performance_req)).detach().numpy()


        scale_back_predict_parameter, _ = inverse_transform(predict_parameter, norm_performance_req, scaler)
        _, simulate_performance = simulator.runSimulation(scale_back_predict_parameter, train=False)

        print("Circuit parameter list is", circuit_config["parameter_list"])
        print("Predict parameter is", scale_back_predict_parameter)
        print("The predicted parameter performance ", simulate_performance)

        continue_key = input("Press Q to quit and other keys to continue: \n")
        if continue_key == "Q":
            return


def inverse_transform(parameter, performance, scaler):
    """
    Inverse transform the data to the original scale
    """
    data = np.hstack((parameter, performance))
    data = scaler.inverse_transform(data)
    return data[:, :parameter.shape[1]], data[:, parameter.shape[1]:]


def train_pytorch_model(model, dataset, device="cpu"):
    train_loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())
    epochs = 30
    train_dataloader = DataLoader(dataset, batch_size=100)
    print("Start Pretraining")

    for epoch in range(epochs):
        model.train()
        avg_loss = 0
        for t, (x, y) in enumerate(train_dataloader):
            # Zero your gradient
            optimizer.zero_grad()
            x_var = torch.autograd.Variable(x.type(torch.FloatTensor)).to(device)
            y_var = torch.autograd.Variable(y.type(torch.FloatTensor).float()).to(device)
            scores = model(x_var)
            loss = train_loss(scores.float(), y_var.float())
            loss = torch.clamp(loss, max=500000, min=-500000)
            avg_loss += (loss.item() - avg_loss) / (t + 1)
            loss.backward()
            optimizer.step()
        print("Epochs {}, Loss {}".format(epoch, avg_loss))
    print("Finish Pretraining")




if __name__ == '__main__':
    useability_pipeline()