from circuit import prepare_data
from dataset import *
import os

from models import ModelEvaluator
from simulator import load_simulator
from utils import load_circuit, load_train_config, load_visual_config, load_model_config, getData, \
    validate_config, save_result, check_save_data_status, saveDictToTxt
from metrics import get_margin_error
from eval_model import *
from visualutils import plot_multiple_margin_with_confidence_cross_fold, \
    plot_multiple_loss_with_confidence_cross_fold, plot_multiple_accuracy_with_confidence_cross_fold
from datetime import datetime
import time

def generate_dataset_given_config(train_config, circuit_config):
    if train_config["pipeline"] == "LourencoPipeline":
        print("Return Lorunco Dataset")
        return LorencoDataset(circuit_config["order"], circuit_config["sign"], train_config["n"], train_config["K"])
    else:
        if train_config["basedata"]:
            print("Return Base Dataset")
            return BaseDataset(circuit_config["order"], circuit_config["sign"])
        else:
            if train_config["subfeasible"]:
                if train_config["duplication"] == 0:
                    print("Return SoftArgMax Dataset")
                    return SoftArgMaxDataset(circuit_config["order"], circuit_config["sign"])
                else:
                    print("Return Ablation Duplication Dataset")
                    return AblationDuplicateDataset(circuit_config["order"], circuit_config["sign"], train_config["duplication"])
            else:
                print("Return Argmax Dataset")
                return ArgMaxDataset(circuit_config["order"], circuit_config["sign"])


def generate_circuit_given_config(train_config):
    config_path = os.path.join(os.path.join(os.getcwd(), "config"), "circuits")
    circuit_mapping = {
        "nmos": os.path.join(os.path.join(config_path, "nmos"), "nmos.yaml"),
        "cascode": os.path.join(os.path.join(config_path, "cascode"), "cascode.yaml"),
        "lna": os.path.join(os.path.join(config_path, "LNA"), "LNA.yaml"),
        "mixer": os.path.join(os.path.join(config_path, "mixer"), "mixer.yaml"),
        "two_stage": os.path.join(os.path.join(config_path, "two_stage"), "two_stage.yaml"),
        "vco": os.path.join(os.path.join(config_path, "VCO"), "VCO.yaml")
    }

    if train_config["circuit"].lower() in circuit_mapping:
        circuit_definition_path = circuit_mapping[train_config["circuit"].lower()]
    else:
        raise KeyError("The circuit you defined does not exist")

    circuit = load_circuit(circuit_definition_path)
    return circuit

def generate_model_given_config(model_config):
    model_mapping = {
        "RandomForestRegressor": RandomForestRegressor,
        "Model500GELU": Model500GELU,
    }

    if model_config["model"] in model_mapping.keys():
        eval_model = model_mapping[model_config["model"]]
        copy_model_config = dict(model_config)
        del copy_model_config["model"]
        return eval_model(**copy_model_config)

    else:
        raise KeyError("The model you defined does not exist")


def generate_visual_given_result(result, train_config, visual_config, pipeline_save_name):
    folder_path = os.path.join(os.path.join(os.getcwd(), "out_plot"), pipeline_save_name)
    os.mkdir(folder_path)
    result_dict = dict()

    if train_config["test_margin_accuracy"] or train_config["train_margin_accuracy"]:
        margin_plot_result = plot_multiple_margin_with_confidence_cross_fold(train_config, visual_config, result, pipeline_save_name)
        result_dict.update(margin_plot_result)
    if train_config["test_accuracy_per_epoch"] or train_config["train_accuracy_per_epoch"]:
        accuracy_plot_result = plot_multiple_accuracy_with_confidence_cross_fold(train_config, visual_config, result, pipeline_save_name)
        result_dict.update(accuracy_plot_result)
    if train_config["loss_per_epoch"]:
        loss_plot_result = plot_multiple_loss_with_confidence_cross_fold(train_config, visual_config, result, pipeline_save_name)
        result_dict.update(loss_plot_result)

    return result_dict

def pipeline():

    train_config = load_train_config()
    validate_config(train_config)
    visual_config = load_visual_config()

    model_config = load_model_config()
    circuit_config = generate_circuit_given_config(train_config)
    dataset = generate_dataset_given_config(train_config, circuit_config)
    simulator = load_simulator(circuit_config)
    model = generate_model_given_config(model_config)


    if train_config["rerun_training"] or not check_save_data_status(circuit_config):
        data_for_evaluation = prepare_data(simulator.parameter_list, simulator.arguments)
        start =time.time()
        print('start sim')
        parameter, performance = simulator.runSimulation(data_for_evaluation, True)
        print('took for sim', time.time()-start)
        print('Params shape', parameter.shape)
        print('Perfomance shape',performance.shape)

        print("Saving metadata for this simulation")
        metadata_path = os.path.join(circuit_config["arguments"]["out"], "metadata.txt")
        saveDictToTxt(circuit_config["arguments"], metadata_path)
    else:
        print("Load from saved data")
        parameter_file_list = [x + ".csv" for x in circuit_config["parameter_list"]]
        performance_file_list = [x + ".csv" for x in circuit_config["performance_list"]]
        parameter, performance = getData(parameter_file_list, performance_file_list, circuit_config["arguments"]["out"])

    print("Pipeline Start")
    pipeline = ModelEvaluator(parameter, performance, dataset, metric=get_margin_error, simulator=simulator,
                              train_config=train_config, model=model)

    cur_time = str(datetime.now().strftime('%Y-%m-%d %H-%M'))
    pipeline_save_name = "{}-circuit-{}-pipeline-{}".format(train_config["circuit"], train_config["pipeline"], cur_time)

    result = pipeline.eval()
    visual_result = generate_visual_given_result(result, train_config, visual_config, pipeline_save_name)
    result.update(visual_result)
    save_result(result, pipeline_save_name)

