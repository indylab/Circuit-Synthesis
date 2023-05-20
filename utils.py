import yaml
import os
from os.path import join
import pandas as pd
import numpy as np
import itertools
import re
from metrics import get_margin_error
from scipy import stats
from torch.cuda import is_available
import platform
from dataset import BaseDataset
import shutil
import time

CONFIG_PATH = os.path.join(os.path.join(os.getcwd(), "config"))

DEFAULT_TRAIN_CONFIG_PATH = os.path.join(CONFIG_PATH, "train_config.yaml")
DEFAULT_VISUAL_CONFIG_PATH = os.path.join(CONFIG_PATH, "visual_config.yaml")
DEFAULT_MODEL_CONFIG_PATH = os.path.join(CONFIG_PATH, "model.yaml")

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

    system = platform.system()
    if system == "Windows":
        config["ngspice_exec"] = config["ngspice_exec_path"]
    else:
        config["ngspice_exec"] = config["ngspice_exec_direct"]

    del config["ngspice_exec_path"]
    del config["ngspice_exec_direct"]
    return config

def load_train_config(configpath=DEFAULT_TRAIN_CONFIG_PATH):

    train_config = load_yaml(configpath)
    if train_config["device"] == "cuda":
        if is_available():
            train_config["device"] = "cuda:0"
        else:
            train_config["device"] = "cpu"

    return train_config

def load_visual_config(configpath=DEFAULT_VISUAL_CONFIG_PATH):
    return load_yaml(configpath)

def load_model_config(configpath=DEFAULT_MODEL_CONFIG_PATH):
    return load_yaml(configpath)


def updateFile(trainingFilePath, outputFilePath, argumentMap,batch_index,path):
    file_name = outputFilePath+f'{batch_index}.sp'
    
    with open(trainingFilePath, 'r') as read_file:
        file_content = read_file.read()
        for key, val in argumentMap.items():
            
            if key == 'out':
                val = path
            
            temp_pattern = "{" + str(key) + "}"
            file_content = file_content.replace(temp_pattern, str(val))
        with open(file_name, 'w') as write_file:
            write_file.write(file_content)
    return file_name


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
    dirs = os.listdir(out)
    for dir in dirs:
        if not(dir.startswith("batch")):
            continue
        try:
            shutil.rmtree(os.path.join(out, dir))
        except PermissionError:
            time.sleep(5)
            shutil.rmtree(os.path.join(out, dir))

def generate_metrics_given_config(train_config):

    metrics_dict = dict()
    if train_config["loss_per_epoch"]:
        metrics_dict["train_loss"] = []
        metrics_dict["validation_loss"] = []
    if train_config["train_accuracy_per_epoch"]:
        metrics_dict["train_accuracy_per_epoch"] = []
    if train_config["test_accuracy_per_epoch"]:
        metrics_dict["validation_accuracy_per_epoch"] = []
    if train_config["test_margin_accuracy"]:
        metrics_dict["test_margins"] = []
    if train_config["train_margin_accuracy"]:
        metrics_dict["train_margins"] = []
    metrics_dict["circuit_error_average"] = []
    metrics_dict["performance_error_average"] = []
    metrics_dict["circuit_error_std"] = []
    metrics_dict["performance_error_std"] = []


    return metrics_dict

def merge_metrics(parent_metrics, child_metrics):

    for k in parent_metrics.keys():
        parent_metrics[k].append(child_metrics[k])


def run_simulation_given_parameter(simulator, parameter_preds, train=False):
    return simulator.runSimulation(parameter_preds, train=train)

def generate_performance_diff_metrics(performance_prediction, test_performance, simulator, train=False):
    margin_error = get_margin_error(performance_prediction, test_performance, simulator.sign)
    metrics_dict = dict()
    if train:
        metrics_dict["train_margins"] = np.max(margin_error, axis=1)
    else:
        metrics_dict["test_margins"] = np.max(margin_error, axis=1)
        metrics_dict["circuit_error_average"] = np.average(margin_error)
        metrics_dict["performance_error_average"] = np.average(margin_error, axis=0)
        metrics_dict["circuit_error_std"] = stats.sem(margin_error)
        metrics_dict["performance_error_std"] = stats.sem(margin_error, axis=0)

    return metrics_dict

def save_result(result, pipeline_save_name, config_path=None):

    save_folder = os.path.join(os.path.join(os.getcwd(), "result_out"), pipeline_save_name)
    os.mkdir(save_folder)
    for k in result.keys():
        out_variable_save_path = os.path.join(save_folder, k + ".npy")
        np.save(out_variable_save_path, result[k])

    if config_path is not None:
        shutil.copyfile(config_path, os.path.join(save_folder, "train_config.yaml"))


def check_save_data_status(circuit_config):
    save_path = circuit_config["arguments"]["out"]

    metadata_path = os.path.join(save_path, "metadata.txt")
    if not os.path.exists(metadata_path):
        print("metadata not exist")
        return False
    else:
        return_dict = parsetxtToDict(metadata_path)
        keys = set(list(circuit_config["arguments"].keys()) + list(return_dict.keys()))

        for key in keys:
            try:
                if circuit_config["arguments"][key] != return_dict[key]:
                    print("Train Config Not Match")
                    return False
            except KeyError:
                print("Train Config Not Match, Additional Parameter")
                return False

    return True

def parsetxtToDict(file_path):
    with open(file_path, "r") as file:
        file_info = file.readlines()
        return_dict = dict()

        for line in file_info:
            line_info = line.strip().split(":")
            try:
                return_dict[line_info[0]] = float(line_info[1])
            except ValueError:
                return_dict[line_info[0]] = line_info[1]
        return return_dict

def saveDictToTxt(dict, save_path):
    with open(save_path, "w") as file:
        count = 0
        for k,v in dict.items():
            if count != 0:
                file.write('\n')
            file.write(str(k) + ":" + str(v))
            count += 1


def sortVector(parameter, performance):
    data = np.hstack((performance, parameter))

    for i in range(len(performance.shape)):
        data = sorted(data, key=lambda x: x[i], reverse=True)
    data = np.array(data)
    return data[:, performance.shape[1]:], data[:, :performance.shape[1]]


def checkAlias(parameter, performance):

    sort_parameter, sort_performance = sortVector(parameter, performance)

    counter = 0
    duplicate_amount = 0
    while counter <= len(sort_performance) - 2:
        if np.all(np.equal(sort_performance[counter], sort_performance[counter + 1])):
            print("BELOW ARE THE DUPLICATE CASE")
            print("THE TWO DIFFERENT PARAMETER")
            print(sort_parameter[counter])
            print(sort_parameter[counter + 1])
            print("THE SAME RESULT PERFORMANCE")
            print(sort_performance[counter])

            duplicate_amount += 1
        counter += 1

    print("TOTAL DUPLICATE CASE IS {}".format(duplicate_amount))
    if duplicate_amount > 0:
        raise ValueError("THERE ARE ALIASING IN THE RESULT")

def evalCircuit(num_sample_check, simulator, scaler, random_scale):

    num_parameter = len(simulator.parameter_list)
    num_performance = len(simulator.performance_list)

    random_parameter = np.random.uniform(-1 * random_scale,random_scale,size=(num_sample_check, num_parameter))
    random_performance = np.random.uniform(-1 * random_scale, random_scale,size=(num_sample_check, num_performance))

    scale_back_parameter, _ = BaseDataset.inverse_transform(random_parameter, random_performance, scaler)

    simulate_parameter, simulate_performance = simulator.runSimulation(scale_back_parameter, train=False)

    print("The parameter been sampled is")
    print(simulate_parameter)
    print("The simulated performance is")
    print(simulate_performance)


def generate_train_config_for_single_pipeline(train_config, model_config, dataset_config):

    new_train_config = dict(train_config)

    del new_train_config["circuits"]
    del new_train_config["dataset"]
    del new_train_config["model_config"]

    if "extra_args" in model_config.keys():
        for k,v in model_config["extra_args"].items():
            new_train_config[k] = v

    for k,v in dataset_config.items():
        new_train_config[k] = v

    return new_train_config

def update_train_config_given_model_type(model_type, train_config):


    train_config["train_margin_accuracy"] = False if "train_margin_accuracy" not in train_config else train_config["train_margin_accuracy"]
    train_config["test_margin_accuracy"] = True if "test_margin_accuracy" not in train_config else train_config["test_margin_accuracy"]
    train_config["loss_per_epoch"] = True if "loss_per_epoch" not in train_config else train_config["loss_per_epoch"]
    train_config["test_accuracy_per_epoch"] = True if "test_accuracy_per_epoch" not in train_config else train_config["test_accuracy_per_epoch"]
    train_config["train_accuracy_per_epoch"] = False if "train_accuracy_per_epoch" not in train_config else train_config["train_accuracy_per_epoch"]

    if model_type == 0:
        #Sklearn model, so no loss and accuracy per epochs
        train_config["loss_per_epoch"] = False
        train_config["test_accuracy_per_epoch"] = False
        train_config["train_accuracy_per_epoch"] = False
    elif model_type == 1:
        #Pytorch model, so have loss, accuracy per epochs if want
        train_config["check_every"] = 20 if "check_every" not in train_config else train_config["check_every"]
        train_config["epochs"] = 100 if "epochs" not in train_config else train_config["epochs"]
        train_config["first_eval"] = 1 if "first_eval" not in train_config else train_config["first_eval"]
        train_config["accuracy_per_epoch_threshold"] = 0.05 if "accuracy_per_epoch_threshold" not in train_config \
            else train_config["accuracy_per_epoch_threshold"]
    else:
        #Lookup model, no loss accuracy and no train margin
        train_config["train_margin_accuracy"] = False
        train_config["loss_per_epoch"] = False
        train_config["test_accuracy_per_epoch"] = False
        train_config["train_accuracy_per_epoch"] = False


def check_comparison_value_diff(train_config, value, key):
    if value is None:
        if key in train_config.keys():
            return train_config[key]
        else:
            return None
    else:
        if key not in train_config.keys() or train_config[key] != value:
            raise ValueError("The {} across different comparison is not the same".format(key))
        else:
            return value


def generate_margin_eval_accuracy_given_config(margin_errors, train_config, visual_config):

    multi_mean = []
    multi_lower_bound = []
    multi_upper_bound = []

    eval_margin = visual_config["margin_threshold"]

    for index, percentage_performance in enumerate(margin_errors):
        temp_mean = []
        temp_lower_bound = []
        temp_upper_bound = []
        for margin in eval_margin:
            temp_run_result = []
            for run in range(len(percentage_performance)):
                if train_config["subset_parameter_check"]:
                    inner_run_performance = percentage_performance[run][index]
                else:
                    inner_run_performance = percentage_performance[run][0]
                greater_num = 0
                for i in inner_run_performance:
                    if i <= margin:
                        greater_num += 1
                temp_run_result.append(greater_num / len(inner_run_performance))

            success = np.array(temp_run_result)
            success_mean = np.average(success)
            success_std = stats.sem(success)

            temp_mean.append(success_mean)
            temp_lower_bound.append(success_mean - success_std)
            temp_upper_bound.append(success_mean + success_std)
        multi_mean.append(temp_mean)
        multi_lower_bound.append(temp_lower_bound)
        multi_upper_bound.append(temp_upper_bound)
    return multi_mean, multi_upper_bound, multi_lower_bound