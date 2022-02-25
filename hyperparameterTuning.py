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
import sys
import os

def normalize(data):
    data_min = np.min(data)
    data_max = np.max(data)
    norm_data = (data-data_min)/(data_max-data_min)
    return norm_data, data_min, data_max

def denormalize(data,data_min,data_max):
    denorm_data = (data * (data_max - data_min) + data_min)
    return denorm_data
def training_function(config, checkpoint_dir=None):
    # Hyperparameters
    test_model = config["model"]()

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
    loss_fn = config["loss_fn"]
    optimizer = config["optim"](test_model.parameters(), lr=config["lr"])  

    # Create Data Loaders
    train_data = DataLoader(train_dataset, batch_size=config["bs"])
    validation_data = DataLoader(val_dataset, batch_size=config["bs"])
    
    # train
    losses, accs, part_accs = train.trainWValidation(test_model, train_data, validation_data, loss_fn, optimizer, num_epochs=int(config["epochs"]), print_every=10,wTune=True,margin=.05)
    
    if checkpoint_dir:
        tune.checkpoint_dir() # don't think this works yet


if __name__ == '__main__':
    '''USAGE:
        use createConfig to set experiment parameters for hyperparametertunining.py 
        Allows multiple experiments to be run simultaneously.
        Pass resulting file to this program (hyperparametertunining.py) as the first and only argument 
    '''
    params = utils.openPickle(sys.argv[1])
    experiment_name = params["experiment_name"]
    dtype = torch.FloatTensor
    loss_functions = [nn.L1Loss().type(dtype)]
    optimizers = [optim.Adagrad]
    
    models = [model.Model500GELU]

    
    # Add a custom metric column, in addition to the default metrics.
    # Note that this must be a metric that is returned in your training results.
    reporter = CLIReporter(
    parameter_columns=["loss_fn", "lr", "epochs", "optim"],
    metric_columns=["accuracy", "validation_accuracy", "partial_accuracy", "validation_partial_accuracy", "loss" ])
    
    analysis = tune.run(
    training_function,
    config = params["config"],
    num_samples = params["num_samples"],
    progress_reporter=reporter,
    keep_checkpoints_num=3, 
    checkpoint_score_attr="validation_accuracy",
    name=experiment_name)

    print("Best config: ", analysis.get_best_config(
        metric="validation_accuracy", mode="max"))

    # Get a dataframe for analyzing trial results.
    filename = f"Experiments/{experiment_name}"
    outfile = open(filename,'wb')

    # store dataframe as pickle in "Experiments/{experiment_name}"
    df = analysis.results_df
    pickle.dump(df,outfile)
    outfile.close()
    