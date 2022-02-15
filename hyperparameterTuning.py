import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Training import model, utils, dataset, train
import numpy as np
from ray import tune
from ray.tune import CLIReporter
import pickle

def normalize(data):
    data_min = np.min(data)
    data_max = np.max(data)
    norm_data = (data-data_min)/(data_max-data_min)
    return norm_data, data_min, data_max

def denormalize(data,data_min,data_max):
    denorm_data = (data * (data_max - data_min) + data_min)
    return denorm_data
def training_function(config):
    # Hyperparameters
    test_model = model.CSGainAndBandwidthManually()

    # load datasets and split into train and val sets
    data = np.array(utils.parseGainAndBWCsv('/home/reevest/Circuit-Synthesis/Data/BW_Gain2.0.csv')).astype(float)
    #data1 = np.array(utils.parseGainAndBWCsv('Data/BW_Gain2.0.csv')).astype(float)
    
    data = data.reshape(data.shape[0],4)
    feature = data.T[0]
    
    # normalize data
    min_max = dict()
    for i in range(4):
        feature = data.T[i]
        norm_feature, data_min, data_max = normalize(feature)
        data.T[i] = norm_feature
        min_max[i] = [data_min, data_max]
    data = data.reshape(data.shape[0],2,2)
    
    # Create data set and split
    dataset1 = dataset.CircuitSynthesisGainAndBandwidthManually(data[:, 1], data[:, 0])
    train_dataset, val_dataset = utils.splitDataset(dataset1, 0.8)
    
    
    # set loss and optimizer
    dtype = torch.FloatTensor
    loss_fn = config["loss_fn"]#nn.L1Loss().type(dtype)  # loss can be changed here. This is the first one i tried that gave meaningful results
    optimizer = config["optim"](test_model.parameters(), lr=config["lr"])  # TODO haven't experimented with this yet

    # Create Data Loaders
    train_data = DataLoader(train_dataset, batch_size=config["bs"])
    validation_data = DataLoader(val_dataset, batch_size=config["bs"])
    
    # train
    losses = train.trainWValidation(test_model, train_data, validation_data, loss_fn, optimizer, num_epochs=int(config["epochs"]), print_every=10,wTune=True,margin=.1)
    
    # return final acc
    acc, part_acc, preds = train.check_accuracy(test_model, validation_data, .1)
    return {"acc": acc, "part_acc": part_acc, "loss": losses[-1]}
    # tune.report(acc=1)
    # tune.report(loss=losses[-1])


if __name__ == '__main__':
    # config = {"lr": 1e-6, "epochs": 100, "loss_fn": nn.L1Loss().type(float)}
    # training_function(config)
    experiment_name = "Feb-14-fix-uniform-2"
    dtype = torch.FloatTensor
    loss_functions = [nn.L1Loss().type(dtype),
                      nn.MSELoss().type(dtype),
                      nn.HuberLoss(reduction='mean', delta=1.0)]
    optimizers = [optim.Adagrad, optim.SGD,
                 optim.Adam]

    
    # Add a custom metric column, in addition to the default metrics.
    # Note that this must be a metric that is returned in your training results.
    reporter = CLIReporter(
    parameter_columns=["loss_fn", "lr", "epochs", "optim"],
    metric_columns=["loss", "acc", "part_acc"])
    
    analysis = tune.run(
    training_function,
    config={
        "epochs": tune.uniform(100, 4000),
        "lr": tune.uniform(0.000001, 0.1),
        "bs":  tune.grid_search([16,32,62,128]),
        "loss_fn" : tune.grid_search(loss_functions),
        "optim" : tune.grid_search(optimizers)
    },
        num_samples = 20,
    progress_reporter=reporter,
    name=experiment_name)

    

    print("Best config: ", analysis.get_best_config(
        metric="acc", mode="max"))

    # Get a dataframe for analyzing trial results.
    filename = f"Experiments/{experiment_name}"
    outfile = open(filename,'wb')

    df = analysis.results_df
    pickle.dump(df,outfile)
    outfile.close()
    