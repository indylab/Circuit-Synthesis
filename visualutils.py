import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats



def plot_multiple_margin_with_confidence_cross_fold(train_config, visual_config, result, save_name):
    #TODO
    pass

def plot_multiple_accuracy_with_confidence_cross_fold(train_config, visual_config, result, save_name):

    result_dict = dict()
    if train_config["train_accuracy_per_epoch"]:
        multi_train_accuracy, multi_train_accuracy_lower_bound, \
        multi_train_accuracy_upper_bound = plot_multiple_accuracy_with_confidence(result["train_accuracy_per_epoch"],
                                                                            train_config, visual_config, save_name + "-train")

        result_dict["multi_train_accuracy"] = multi_train_accuracy
        result_dict["multi_train_accuracy_lower_bound"] = multi_train_accuracy_lower_bound
        result_dict["multi_train_accuracy_upper_bound"] = multi_train_accuracy_upper_bound

    if train_config["test_accuracy_per_epoch"]:
        multi_test_accuracy, multi_test_accuracy_lower_bound, \
        multi_test_accuracy_upper_bound = plot_multiple_accuracy_with_confidence(result["validation_accuracy_per_epoch"],
                                                                            train_config, visual_config, save_name + "-test")
        result_dict["multi_test_accuracy"] = multi_test_accuracy
        result_dict["multi_test_accuracy_lower_bound"] = multi_test_accuracy_lower_bound
        result_dict["multi_test_accuracy_upper_bound"] = multi_test_accuracy_upper_bound

    return result_dict

def plot_multiple_loss_with_confidence_cross_fold(train_config, visual_config, result, save_name):
    multi_train_loss, multi_train_loss_lower_bounds, multi_train_loss_upper_bounds = plot_multiple_loss_with_confidence(result["train_loss"],
                                                                                                                        train_config, visual_config, save_name + "-train")
    multi_test_loss, multi_test_loss_lower_bounds, multi_test_loss_upper_bounds = plot_multiple_loss_with_confidence(result["validation_loss"],
                                                                                                                     train_config, visual_config, save_name + "-test")

    result_dict = dict()
    result_dict["multi_train_loss"] = multi_train_loss
    result_dict["multi_test_loss"] = multi_test_loss
    result_dict["multi_train_loss_lower_bounds"] = multi_train_loss_lower_bounds
    result_dict["multi_test_loss_lower_bounds"] = multi_test_loss_lower_bounds
    result_dict["multi_train_loss_upper_bounds"] = multi_train_loss_upper_bounds
    result_dict["multi_test_loss_upper_bound"] = multi_test_loss_upper_bounds

    return result_dict

def plot_multiple_margin_with_confidence(margin, train_config, visual_config, save_name, lookup_margin=None):
    pass

def plot_multiple_accuracy_with_confidence(accuracy, train_config, visual_config, save_name):

    font_size = visual_config["font_size"]
    plt.rcParams.update({'font.size': font_size})

    epochs = train_config["epochs"]
    check_every = train_config["check_every"]
    first_eval = train_config["first_eval"]
    subset = train_config["subset"]

    step = epochs // check_every
    color = visual_config["color"][:len(subset)]

    save_path = os.path.join(os.path.join(os.getcwd(), "out_plot"), save_name + "-accuracy.png")

    if first_eval is not None:
        eva_epochs = [i * check_every for i in range(step + 1)]
        eva_epochs[0] = first_eval
    else:
        eva_epochs = [(i + 1) * check_every for i in range(step)]

    multi_accuracy = []
    multi_accuracy_lower_bound = []
    multi_accuracy_upper_bound = []

    for percentage_performance in accuracy:
        temp_performance_mean = np.average(percentage_performance, axis=0)
        temp_performance_std = stats.sem(percentage_performance, axis=0)
        multi_accuracy.append(temp_performance_mean)
        multi_accuracy_lower_bound.append(temp_performance_mean - temp_performance_std)
        multi_accuracy_upper_bound.append(temp_performance_mean + temp_performance_std)

    fig = plt.figure()
    ax = fig.add_subplot()

    for i in range(len(multi_accuracy)):
        if subset[i] <= 0.5:
            temp_label = "{}% data".format(subset[i] * 100)
        else:
            split_size = np.gcd(int(subset[i] * 100), 100)
            fold = int(100 / split_size)
            temp_label = "{}-fold".format(fold)

        ax.plot(eva_epochs, multi_accuracy[i], label=temp_label, color=color[i])
        ax.fill_between(eva_epochs, multi_accuracy_lower_bound[i],
                        multi_accuracy_upper_bound[i], alpha=.3, color=color[i])


    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    ax.legend()
    plt.ylabel("Test Success Rate")
    plt.xlabel("Epochs")

    plt.savefig(save_path, dpi=250)


    return multi_accuracy, multi_accuracy_lower_bound, multi_accuracy_upper_bound


def plot_multiple_loss_with_confidence(loss, train_config, visual_config, save_name):

    font_size = visual_config["font_size"]
    plt.rcParams.update({'font.size': font_size})
    multi_loss = []
    multi_loss_lower_bounds = []
    multi_loss_upper_bounds = []

    num_subset = len(loss)
    color = visual_config["color"][:num_subset]
    subset = train_config["subset"]
    epochs = train_config["epochs"]

    save_path = os.path.join(os.path.join(os.getcwd(),"out_plot"), save_name + "-loss.png")

    for percentage_loss in loss:
        temp_loss_mean = np.average(percentage_loss, axis=0)
        temp_loss_std = stats.sem(percentage_loss, axis=0)

        multi_loss.append(temp_loss_mean)
        multi_loss_lower_bounds.append(temp_loss_mean - temp_loss_std)
        multi_loss_upper_bounds.append(temp_loss_mean + temp_loss_std)

    fig = plt.figure()
    ax = fig.add_subplot()

    for i in range(len(multi_loss)):
        if subset[i] <= 0.5:
            temp_label = "{}% data".format(subset[i] * 100)
        else:
            split_size = np.gcd(int(subset[i] * 100), 100)
            fold = int(100 / split_size)
            temp_label = "{}-fold".format(fold)

        ax.plot(np.arange(epochs), multi_loss[i], label=temp_label, color=color[i])
        ax.fill_between(np.arange(epochs), multi_loss_lower_bounds[i], multi_loss_upper_bounds[i], alpha=.3, color=color[i])


        ax.set_xlim([0, None])
        ax.set_ylim([0, None])
        ax.legend()
        plt.ylabel("Test {} Loss".format("L1"))
        plt.xlabel("Epochs")

        plt.savefig(save_path, dpi=250)

    return multi_loss, multi_loss_lower_bounds, multi_loss_upper_bounds