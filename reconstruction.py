from utils import load_visual_config, load_train_config, update_train_config_given_model_type
from visualutils import *

def reload_training_raw_data(result_path):

    result_dict = dict()
    for (dirpath, _, filenames) in os.walk(result_path):

        for file in filenames:
            file_info = file.split(".")[0]
            file_extention = file.split(".")[1]
            if file_extention == "npy":
                result_dict[file_info] = np.load(os.path.join(dirpath, file), allow_pickle=True)

    return result_dict


def reconstruct_comparing_plot(result_paths, new_save_path, train_config = None):

    '''
    Result_paths is the list of folder where you store all the raw data and want to do the plot comparison. However remember the order of the result
    match the train_config
    New save path is the new folder name where you want to store the new reconstructed plot
    If the data you want to reconstruct is run before this code change, you have to supply your custom train_config, otherwise leave it as None
    '''

    visual_config = load_visual_config()
    if train_config is None:
        for path in result_paths:
            if os.path.exists(os.path.join(path, "train_config.yaml")):
                train_config = load_train_config(os.path.join(path, "train_config.yaml"))

    new_save_path_folder = os.path.join(os.getcwd(), "out_plot", new_save_path)
    os.mkdir(new_save_path_folder)

    result_dict_list = []
    for path in result_paths:
        result_dict_list.append(reload_training_raw_data(path))

    if train_config["compare_dataset"]:
        labels = [i["type"] for i in train_config["dataset"]]
    elif train_config["compare_method"]:
        labels = [i["model"] for i in train_config["model_config"]]
    else:
        raise ValueError

    epochs = 100 if "epochs" not in train_config else train_config["epochs"]
    check_every = 20 if "check_every" not in train_config else train_config["check_every"]
    first_eval = 1 if "first_eval" not in train_config else train_config["first_eval"]

    if train_config["compare_dataset"] or train_config["compare_method"]:
        if "multi_train_loss" in result_dict_list[0].keys():

            multi_loss = [i["multi_train_loss"] for i in result_dict_list]
            multi_loss_upper_bound = [i["multi_train_loss_upper_bound"] for i in result_dict_list]
            multi_loss_lower_bound = [i["multi_train_loss_lower_bound"] for i in result_dict_list]
            plot_multiple_loss_with_confidence_comparison(multi_loss, multi_loss_upper_bound, multi_loss_lower_bound,
                                                          labels, train_config["subset"], new_save_path_folder, visual_config, epochs)

        if "multi_test_accuracy" in result_dict_list[0].keys():
            multi_accuracy = [i["multi_test_accuracy"] for i in result_dict_list]
            multi_accuracy_upper_bound = [i["multi_test_accuracy_upper_bound"] for i in result_dict_list]
            multi_accuracy_lower_bound = [[i["multi_test_accuracy_lower_bound"] for i in result_dict_list]]
            plot_multiple_accuracy_per_epochs_with_confidence_comparison(multi_accuracy, multi_accuracy_upper_bound, multi_accuracy_lower_bound,
                                                                         labels, train_config["subset"], new_save_path_folder,
                                                                         visual_config, epochs, check_every, first_eval)

        if "multi_test_mean" in result_dict_list[0].keys():
            multi_margin = [i["multi_test_mean"] for i in result_dict_list]
            multi_margin_upper_bound = [i["multi_test_upper_bound"] for i in result_dict_list]
            multi_margin_lower_bound = [i["multi_test_lower_bound"] for i in result_dict_list]
            plot_multiple_margin_with_confidence_comparison(multi_margin, multi_margin_upper_bound,
                                                            multi_margin_lower_bound, labels, train_config["subset"], new_save_path_folder, visual_config)



def reconstruct_plot(result_path, new_save_path, train_config = None):
    '''
    Result_path is the folder where you store all the raw data
    New save path is the new folder name where you want to store the new reconstructed plot
    If the data you want to reconstruct is run before this code change, you have to supply your custom train_config, otherwise leave it as None
    '''


    visual_config = load_visual_config()

    if train_config is None:
        train_config = load_train_config(os.path.join(result_path, "train_config.yaml"))

    result_dict = reload_training_raw_data(result_path)

    new_save_path_folder = os.path.join(os.getcwd(), "out_plot", new_save_path)
    os.mkdir(new_save_path_folder)

    if "multi_train_loss" in result_dict.keys():
        plot_loss(result_dict["multi_train_loss"], result_dict["multi_train_loss_upper_bound"],
                  result_dict["multi_train_loss_lower_bound"], visual_config, train_config, save_name=os.path.join(new_save_path_folder, "loss.png"))

    if "multi_test_accuracy" in result_dict.keys():
        plot_accuracy(result_dict["multi_test_accuracy"], result_dict["multi_test_accuracy_upper_bound"],
                      result_dict["multi_test_accuracy_lower_bound"], visual_config, train_config, save_name=os.path.join(new_save_path_folder, "accuracy.png"))

    if "multi_test_mean" in result_dict.keys():
        plot_margin(result_dict["multi_test_mean"], result_dict["multi_test_upper_bound"],
                    result_dict["multi_test_lower_bound"], visual_config, train_config, save_name=os.path.join(new_save_path_folder, "margin.png"))




