import yaml
import os
from os.path import join
import pandas as pd
import numpy as np
from time import time, sleep
import itertools
import re

CONFIG_PATH = os.path.join(os.path.join(os.getcwd(), "config"))

DEFAULT_TRAIN_CONFIG_PATH = os.path.join(CONFIG_PATH, "train_config.yaml")
DEFAULT_VISUAL_CONFIG_PATH = os.path.join(CONFIG_PATH, "visual_config.yaml")
DEFAULT_MODEL_CONFIG_PATH = os.path.join(CONFIG_PATH, "model.yaml")
DEFAULT_CONFIG_CONFLICT_PATH = os.path.join(CONFIG_PATH, "config_conflict.yaml")

def load_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        config_yaml = yaml.safe_load(file)
    return config_yaml

def load_circuit(circuit_name):
    """Load circuit"""
    config_folder = join("config", "circuits")
    folder = join(config_folder, circuit_name)
    config = load_yaml(folder)

    ##Loading ngspice config
    folder = join('config', 'ngspice.yaml')
    config_ngspice = load_yaml(folder)

    config.update(config_ngspice)
    return config

def load_train_config(configpath=DEFAULT_TRAIN_CONFIG_PATH):

    train_config = load_yaml(configpath)
    default_config_folder_path = os.path.join(CONFIG_PATH, "default_config")
    existing_pipeline = {
        "CrossFoldValidationPipeline": os.path.join(default_config_folder_path, "crossfold_validation_train_config.yaml"),
        "LourencoPipeline": os.path.join(default_config_folder_path, "Lourenco_train_config.yaml"),
        "SklearnPipeline": os.path.join(default_config_folder_path, "sklearn_train_config.yaml")
    }

    if train_config["pipeline"] in existing_pipeline.keys():
        default_config_path = existing_pipeline[train_config["pipeline"]]
        default_config = load_yaml(default_config_path)
        for k in default_config.keys():
            if k not in train_config:
                train_config[k] = default_config[k]
    else:
        raise KeyError("The Pipeline you specify is not defined")

    return train_config

def load_visual_config(configpath=DEFAULT_VISUAL_CONFIG_PATH):
    return load_yaml(configpath)

def load_model_config(configpath=DEFAULT_MODEL_CONFIG_PATH):
    return load_yaml(configpath)

def load_conflict_config(configpath=DEFAULT_CONFIG_CONFLICT_PATH):
    return load_yaml(configpath)

def updateFile(trainingFilePath, outputFilePath, argumentMap):

    with open(trainingFilePath, 'r') as read_file:
        file_content = read_file.read()
        for key, val in argumentMap.items():
            temp_pattern = "{" + str(key) + "}"
            file_content = file_content.replace(temp_pattern, str(val))

        with open(outputFilePath, 'w') as write_file:
            write_file.write(file_content)


def convert2numpy(filenames):
    files = []
    for file in filenames:
        file_data = pd.read_csv(file, header=None)
        file_data = file_data.apply(lambda x: re.split(r"\s+", str(x).replace("=", ""))[2], axis=1)
        files.append(file_data)

    combine = pd.concat(files, axis=1)
    return np.array(combine, dtype=float)


def getData(param_outfile_names, perform_outfile_names, out):
    param_fullname = [os.path.join(out, file) for file in param_outfile_names]
    perform_fullname = [os.path.join(out, file) for file in perform_outfile_names]
    x = convert2numpy(param_fullname)
    y = convert2numpy(perform_fullname)
    return x, y


def delete_testing_files(out_directory, names):
    out = out_directory
    names = list(itertools.chain(*names))
    files = [os.path.join(out, file) for file in names]

    for file in files:
        try:
            os.remove(file)
        except FileNotFoundError:
            continue
        except PermissionError:
            sleep(1)
            os.remove(file)

def validate_config(train_config):
    conflict_config = load_conflict_config()
    for rule in conflict_config["conflict_rule"]:
        if rule["pipeline"] == train_config["pipeline"]:
            if rule["type"] == "Boolean":
                key = rule["key"]
                value = rule["value"]
                assert train_config[key] == value, "The {} config you choose for {} have conflict, " \
                                                   "please fix the conflict or use default value".format(key, rule["pipeline"])
