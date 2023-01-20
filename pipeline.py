from circuit import prepare_data
from dataset import *
import os

from models import ModelEvaluator
from simulator import load_simulator
from utils import load_circuit, load_train_config, load_visual_config, \
    save_result, check_save_data_status, saveDictToTxt, checkAlias, \
    generate_train_config_for_single_pipeline, update_train_config_given_model_type, check_comparison_value_diff
from metrics import get_margin_error, get_relative_margin_error
from eval_model import *
from visualutils import plot_multiple_margin_with_confidence_entrypoint, \
    plot_multiple_loss_with_confidence_entrypoint, plot_multiple_accuracy_with_confidence_entrypoint, \
    plot_multiple_margin_with_confidence_comparison, plot_multiple_loss_with_confidence_comparison, \
    plot_multiple_accuracy_per_epochs_with_confidence_comparison, \
    plot_multiple_subset_parameter_margin_accuracy_with_confidence_entrypoint
from datetime import datetime
import time

def generate_dataset_given_config(train_config, circuit_config, dataset_config):
    epsilon = train_config["epsilon"]
    dataset_type = dataset_config["type"]
    subset_parameter_mode = train_config["mode"]

    if subset_parameter_mode not in ("drop", "replace"):
        raise ValueError("Provided Parameter argmax replace policy is not defined")

    if dataset_type == "Lourenco":
        print("Return Lourenco Dataset")
        dataset_config["n"] = 0.15 if "n" not in dataset_config else dataset_config["n"]
        dataset_config["K"] = 15 if "K" not in dataset_config else dataset_config["K"]

        return LorencoDataset(circuit_config["order"], circuit_config["sign"], dataset_config["n"], dataset_config["K"], dataset_config, epsilon)

    if dataset_type =="Base":
        print("Return Base Dataset")
        return BaseDataset(circuit_config["order"], circuit_config["sign"], dataset_config)

    if dataset_type =='SoftArgmax':
        print("Return SoftArgMax Dataset")

        return SoftArgMaxDataset(circuit_config["order"], circuit_config["sign"],
                                 dataset_config, epsilon, subset_parameter_mode)

    if dataset_type =='SoftBase':
        print("Return Soft Base Dataset")
        return SoftBaseDataset(circuit_config["order"], circuit_config["sign"], dataset_config, epsilon)

    if dataset_type =='Ablation':
        print("Return Ablation Duplication Dataset")
        dataset_config["duplication"] = 20 if "duplication" not in dataset_config else dataset_config["duplication"]
        return AblationDuplicateDataset(circuit_config["order"], circuit_config["sign"],
                                        dataset_config["duplication"],
                                        dataset_config, epsilon, subset_parameter_mode)

    if dataset_type =='Argmax':
        print("Return Argmax Dataset")
        return ArgMaxDataset(circuit_config["order"], circuit_config["sign"],
                             dataset_config, epsilon, subset_parameter_mode)


def generate_circuit_given_config(circuit_name):
    config_path = os.path.join(os.path.join(os.getcwd(), "config"), "circuits")
    circuit_mapping = {
        "nmos": os.path.join(os.path.join(config_path, "nmos"), "nmos.yaml"),
        "cascode": os.path.join(os.path.join(config_path, "cascode"), "cascode.yaml"),
        "lna": os.path.join(os.path.join(config_path, "LNA"), "LNA.yaml"),
        "mixer": os.path.join(os.path.join(config_path, "mixer"), "mixer.yaml"),
        "two_stage": os.path.join(os.path.join(config_path, "two_stage"), "two_stage.yaml"),
        "vco": os.path.join(os.path.join(config_path, "VCO"), "VCO.yaml"),
        "pa": os.path.join(os.path.join(config_path, "pa"), "pa.yaml"),
    }

    if circuit_name.lower() in circuit_mapping:
        circuit_definition_path = circuit_mapping[circuit_name.lower()]
    else:
        raise KeyError("The circuit you defined does not exist")

    circuit = load_circuit(circuit_definition_path)
    return circuit

def generate_model_given_config(model_config,num_params,num_perf):

    
    sklearn_model_mapping = {
        "RandomForestRegressor": SklearnModel,

    }

    dl_model_mapping = {
        "Model500GELU": Model500GELU,
    }

    lookup_model_mapping = {
        "Lookup": None
    }

    if model_config["model"] in sklearn_model_mapping.keys():
        eval_model = sklearn_model_mapping[model_config["model"]]
        copy_model_config = dict(model_config)
        copy_model_config.pop("extra_args", None)
        copy_model_config.pop("model", None)
        return eval_model(**copy_model_config), 0
    elif model_config["model"] in dl_model_mapping.keys():
        model_config['parameter_count'] = num_perf
        model_config['output_count'] = num_params
        eval_model = dl_model_mapping[model_config["model"]]
        copy_model_config = dict(model_config)
        copy_model_config.pop("extra_args", None)
        copy_model_config.pop("model", None)
        return eval_model(**copy_model_config), 1
    elif model_config["model"] in lookup_model_mapping.keys():
        return None, 2
    else:
        raise KeyError("The model you defined does not exist")


def generate_visual_given_result(result, train_config, visual_config, pipeline_save_name, dataset_type):
    folder_path = os.path.join(os.path.join(os.getcwd(), "out_plot"), pipeline_save_name)
    try:
        os.mkdir(folder_path)
    except:
        pass #if less than a minute passed
    result_dict = dict()

    if train_config["test_margin_accuracy"] or train_config["train_margin_accuracy"]:
        margin_plot_result = plot_multiple_margin_with_confidence_entrypoint(train_config, visual_config, result, pipeline_save_name, dataset_type)
        result_dict.update(margin_plot_result)
    if train_config["test_accuracy_per_epoch"] or train_config["train_accuracy_per_epoch"]:
        accuracy_plot_result = plot_multiple_accuracy_with_confidence_entrypoint(train_config, visual_config, result, pipeline_save_name)
        result_dict.update(accuracy_plot_result)
    if train_config["loss_per_epoch"]:
        loss_plot_result = plot_multiple_loss_with_confidence_entrypoint(train_config, visual_config, result, pipeline_save_name)
        result_dict.update(loss_plot_result)
    if train_config["subset_parameter_check"]:
        subset_parameter_plot_result = plot_multiple_subset_parameter_margin_accuracy_with_confidence_entrypoint(train_config,
                                                                                                                 visual_config, result, pipeline_save_name)
        result_dict.update(subset_parameter_plot_result)
    return result_dict


def generate_circuit_status(circuit_config, parameter, performance, train_config, path):

    circuit_dict = dict()
    circuit_dict["num_parameter"] = parameter.shape[1]
    circuit_dict["num_performance"] = performance.shape[1]
    circuit_dict["data_size"] = performance.shape[0]

    argmax_dataset = ArgMaxDataset(circuit_config["order"], circuit_config["sign"], train_config)

    _, _, extra_info = argmax_dataset.modify_data(parameter, performance, None, None, True)

    circuit_dict["argmax_ratio"] = extra_info["Argmax_ratio"]
    circuit_dict["argmax_modify_num"] = extra_info["Argmax_modify_num"]

    saveDictToTxt(circuit_dict, path)



def pipeline(configpath):

    train_config = load_train_config(configpath=configpath)
    visual_config = load_visual_config()


    if train_config["compare_dataset"] and train_config["compare_method"]:
        raise ValueError("You cannot compare dataset and method at the same time")

    if (train_config["compare_dataset"] or train_config["compare_method"]) and \
            (len(train_config["model_config"]) > 1 and len(train_config["dataset"]) > 1):
        raise ValueError("When you doing comparison testing, dataset and model can not be both greater than 1")

    for circuit in train_config['circuits']:
        print("Pipeline with {} circuit".format(circuit))
        pipeline_cur_time = str(datetime.now().strftime('%Y-%m-%d %H-%M'))
        if train_config["compare_dataset"]:
            save_path = os.path.join(os.getcwd(), "out_plot", pipeline_cur_time + "-" + "compare-dataset-" + circuit)
        else:
            save_path = os.path.join(os.getcwd(), "out_plot", pipeline_cur_time + "-" + "compare-method-" + circuit)
        print("Save comparison folder is {}".format(save_path))

        compare_margin_error_mean_list = []
        compare_margin_error_upper_bound_list = []
        compare_margin_error_lower_bound_list = []

        compare_loss_mean_list = []
        compare_loss_upper_bound_list = []
        compare_loss_lower_bound_list = []

        compare_accuracy_per_epochs_mean_list = []
        compare_accuracy_per_epochs_upper_bound_list = []
        compare_accuracy_per_epochs_lower_bound_list = []

        label = []

        epochs = None
        check_every = None
        first_eval = None
        test_margin_accuracy = None
        loss_per_epoch = None
        test_accuracy_per_epoch = None

        for model_template_config in train_config["model_config"]:
            print("Pipeline with {} model".format(model_template_config["model"]))
            for dataset_type_config in train_config["dataset"]:

                circuit_config = generate_circuit_given_config(circuit)
                dataset = generate_dataset_given_config(train_config, circuit_config, dataset_type_config)

                new_train_config = generate_train_config_for_single_pipeline(train_config, model_template_config, dataset_type_config)

                simulator = load_simulator(circuit_config=circuit_config,
                                            simulator_config=new_train_config['simulator_config'])

                model, model_type = generate_model_given_config(dict(model_template_config),num_params=simulator.num_params,
                                                                 num_perf=simulator.num_perf)

                update_train_config_given_model_type(model_type, new_train_config)
                if train_config["compare_dataset"] or train_config["compare_method"] or dataset_type_config[
                    "type"] not in ("SoftArgmax", "Argmax"):
                    new_train_config["subset_parameter_check"] = False
                new_train_config["model_type"] = model_type
                test_margin_accuracy = check_comparison_value_diff(new_train_config, test_margin_accuracy, "test_margin_accuracy")
                loss_per_epoch = check_comparison_value_diff(new_train_config, loss_per_epoch, "loss_per_epoch")
                test_accuracy_per_epoch = check_comparison_value_diff(new_train_config, test_accuracy_per_epoch, "test_accuracy_per_epoch")

                if new_train_config["test_accuracy_per_epoch"]:
                    epochs = check_comparison_value_diff(new_train_config, epochs, "epochs")
                    check_every = check_comparison_value_diff(new_train_config, check_every, "check_every")
                    first_eval = check_comparison_value_diff(new_train_config, first_eval, "first_eval")
                elif new_train_config["loss_per_epoch"]:
                    epochs = check_comparison_value_diff(new_train_config, epochs, "epochs")

                if new_train_config["rerun_training"] or not check_save_data_status(circuit_config):
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
                    parameter= np.load(os.path.join(simulator.arguments["out"], "x.npy"))
                    performance =np.load(os.path.join(simulator.arguments["out"], "y.npy"))

                print("Check Alias Problem")
                checkAlias(parameter, performance)

                print("Generate Circuit Status")
                circuit_status_path = os.path.join(os.getcwd(), circuit_config["arguments"]["out"], "circuit_stats.txt")
                if not os.path.exists(circuit_status_path):
                    generate_circuit_status(circuit_config, parameter, performance, new_train_config, circuit_status_path)

                print("Pipeline Start")
                if new_train_config["metric"] == "absolute":
                    use_metric = get_margin_error
                else:
                    use_metric = get_relative_margin_error

                model_pipeline = ModelEvaluator(parameter, performance, dataset, metric=use_metric, simulator=simulator,
                                          train_config=new_train_config, model=model)

                cur_time = str(datetime.now().strftime('%Y-%m-%d %H-%M'))
                pipeline_save_name = "{}-circuit-{}-dataset-{}-method-{}".format(circuit,
                                                                                 dataset_type_config["type"], model_template_config["model"], cur_time)
                print("Pipeline save name is {}".format(pipeline_save_name))
                result = model_pipeline.eval()
                visual_result = generate_visual_given_result(result, new_train_config,
                                                             visual_config, pipeline_save_name, dataset_type_config["type"])
                result.update(visual_result)
                save_result(result, pipeline_save_name, configpath)

                if new_train_config["compare_dataset"] or new_train_config["compare_method"]:
                    if new_train_config["loss_per_epoch"]:
                        compare_loss_mean_list.append(result["multi_train_loss"])
                        compare_loss_upper_bound_list.append(result["multi_train_loss_upper_bound"])
                        compare_loss_lower_bound_list.append(result["multi_train_loss_lower_bound"])
                    if new_train_config["test_accuracy_per_epoch"]:
                        compare_accuracy_per_epochs_mean_list.append(result["multi_test_accuracy"])
                        compare_accuracy_per_epochs_upper_bound_list.append(result["multi_test_accuracy_upper_bound"])
                        compare_accuracy_per_epochs_lower_bound_list.append(result["multi_test_accuracy_lower_bound"])
                    if new_train_config["test_margin_accuracy"]:
                        compare_margin_error_mean_list.append(result["multi_test_mean"])
                        compare_margin_error_lower_bound_list.append(result["multi_test_lower_bound"])
                        compare_margin_error_upper_bound_list.append(result["multi_test_upper_bound"])
                if new_train_config["compare_dataset"]:
                    label.append(dataset_type_config["type"])
                if new_train_config["compare_method"]:
                    label.append(model_template_config["model"])

        if train_config["compare_dataset"] or train_config["compare_method"]:
            if test_margin_accuracy:
                plot_multiple_margin_with_confidence_comparison(compare_margin_error_mean_list,
                                                                compare_margin_error_upper_bound_list,
                                                                compare_margin_error_lower_bound_list,
                                                                label, train_config["subset"], save_path, visual_config)
            if loss_per_epoch:
                plot_multiple_loss_with_confidence_comparison(compare_loss_mean_list, compare_loss_upper_bound_list,
                                                              compare_loss_lower_bound_list, label, train_config["subset"],
                                                              save_path, visual_config, epochs)

            if test_accuracy_per_epoch:
                plot_multiple_accuracy_per_epochs_with_confidence_comparison(compare_accuracy_per_epochs_mean_list,
                                                                             compare_accuracy_per_epochs_upper_bound_list,
                                                                             compare_accuracy_per_epochs_lower_bound_list,
                                                                             label, train_config["subset"], save_path,
                                                                             visual_config, epochs, check_every, first_eval)


