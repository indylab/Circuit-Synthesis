from circuit import prepare_data
from dataset import *
import os

from models import ModelEvaluator
from simulator import load_simulator
from utils import load_circuit, load_train_config, load_visual_config, load_model_config, getData, validate_config
from metrics import get_margin_error
from eval_model import *

def generate_dataset_given_config(train_config):
    if train_config["pipeline"] == "LourencoPipeline":
        return LorencoDataset
    else:
        if train_config["subfeasible"]:
            if train_config["duplication"] == 0:
                return SoftArgMaxDataset
            else:
                return AblationDuplicateDataset
        else:
            return ArgMaxDataset


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
        "RandomForestRegressor": RandomForestRegressor
    }

    if model_config["model"] in model_mapping.keys():
        eval_model = model_mapping[model_config["model"]]
        copy_model_config = dict(model_config)
        del copy_model_config["model"]
        return eval_model(**copy_model_config)

    else:
        raise KeyError("The model you defined does not exist")





def pipeline():

    train_config = load_train_config()

    validate_config(train_config)
    visual_config = load_visual_config()
    model_config = load_model_config()
    circuit_config = generate_circuit_given_config(train_config)
    dataset = generate_dataset_given_config(train_config)
    simulator = load_simulator(circuit_config)
    model = generate_model_given_config(model_config)

    if train_config["rerun_training"]:
        data_for_evaluation = prepare_data(simulator.parameter_list, simulator.arguments)
        parameter, performance = simulator.runSimulation(data_for_evaluation, True)
    else:
        parameter_file_list = [x + ".csv" for x in circuit_config["parameter_list"]]
        performance_file_list = [x + ".csv" for x in circuit_config["performance_list"]]
        parameter, performance = getData(parameter_file_list, performance_file_list, circuit_config["arguments"]["out"])


    pipeline = ModelEvaluator(parameter, performance, dataset, metric=get_margin_error, simulator=simulator,
                              train_config=train_config, model=model)

    result = pipeline.eval()

