import numpy as np
from ray import tune
from ray.tune import CLIReporter
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from Training import model, utils, dataset, train



if __name__ == '__main__':
    '''USAGE:
        used to create config and set experiment parameters for hyperparametertunining.py 
        Allows multiple experiments to be run simultaneously.
        Saves file to 'Experiments/configs/{experiment_name}-params'
        pass this file to hyperparametertuning.py as only parameter. 
    '''
    dtype = torch.FloatTensor
    loss_functions = [nn.L1Loss().type(dtype)]
    optimizers = [optim.Adagrad]
    models = [model.Model500GELU]

    config={
            "epochs": 1000,#tune.uniform(500, 1000),
            "lr": tune.uniform(0.0001, 0.05),
            "bs":  5, #tune.randint(1,32),
            "loss_fn" : tune.grid_search(loss_functions),
            "optim" : tune.grid_search(optimizers),
            "model" : tune.grid_search(models)
        }
    experiment_name = "Feb-17-5pct-acc-test"
    num_samples = 25 # IMPORTANT drasticly affects training time. this is how many of each grid search options will be sampled:
                     # if you 5 models and 5 loss functions to try. Ray will make "num_samples" trials for each of the 5 models and "num_samples" trials for each of the 5 loss_functions: ie "num_samples"^2 * 25 trials.
    parameters = dict({"experiment_name" : experiment_name, "num_samples" : num_samples, "config": config})
    filename = f"Experiments/configs/{experiment_name}-params"

    outfile = open(filename,'wb')

    pickle.dump(parameters,outfile)
    outfile.close()