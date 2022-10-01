import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS
import numpy as np
import seaborn as sns
from scipy import stats

def plot_parameter(X, Y, reduce_dim_x = True, reduce_dim_y = True):

    assert X.shape[1] == 2 or reduce_dim_x
    assert Y.shape[1] == 2 or reduce_dim_y

    embedded_X = X
    embedded_Y = Y
    if reduce_dim_x and X.shape[1] > 2:
        #embedded_X = TSNE(n_components=2, init='random').fit_transform(X)
        #embedded_X = LocallyLinearEmbedding(n_components=2).fit_transform(X)
        embedded_X = MDS(n_components=2).fit_transform(X)
    if reduce_dim_y and Y.shape[1] > 2:
        #embedded_Y = TSNE(n_components=2, init='random').fit_transform(Y)
        #embedded_Y = LocallyLinearEmbedding(n_components=2).fit_transform(Y)
        embedded_Y = MDS(n_components=2).fit_transform(Y)

    fig, axes = plt.subplots(figsize=(36, 12))
    axes.get_xaxis().set_visible(False)  # remove erroreas graph axis
    axes.get_yaxis().set_visible(False)

    C = np.ones((X.shape[0], 1, 3))

    for i,c in enumerate(C):
        c[:,1] = embedded_X[i,0] / embedded_X[:, 0].max()
        c[:,2] = embedded_X[i,1] / embedded_X[:, 1].max()
    C = C.reshape(-1, 3)
    fig.add_subplot(121)
    plt.scatter(embedded_X[:, 0], embedded_X[:, 1], c= C)
    fig.add_subplot(122)
    plt.scatter(embedded_Y[:, 0], embedded_Y[:, 1], c= C)
    plt.show()


def graph_multiple_margin_with_confidence_cross_fold(margin_errors, margins, subset,  baseline = None, vertical_point = 0.05,
                                                     percentage = True, std=True, log=True, graph=True, color=None):


    num_percentage = len(margin_errors)
    if color is None:
        color_array = []

        for i in range(num_percentage):
            color_array.append(np.random.rand(3, ))
    else:
        color_array = color

    multi_mean = []
    multi_lower_bound = []
    multi_upper_bound = []

    #Outer axis is different percentage, inner axis is different run, most inner axis is each prediction number
    for percentage_performance in margin_errors:
        temp_mean = []
        temp_lower_bound = []
        temp_upper_bound = []
        for margin in margins:
            temp_run_result = []
            for run in range(len(percentage_performance)):
                inner_run_performance = percentage_performance[run]
                greater_num = 0
                for i in inner_run_performance:
                    if i <= margin:
                        greater_num += 1
                if percentage:
                    temp_run_result.append(greater_num / len(inner_run_performance))
                else:
                    temp_run_result.append(greater_num)
            success = np.array(temp_run_result)
            success_mean = np.average(success)
            if std:
                success_var = stats.sem(success)
            else:
                success_var = np.var(success)
            temp_mean.append(success_mean)
            temp_lower_bound.append(success_mean - success_var)
            temp_upper_bound.append(success_mean + success_var)
        multi_mean.append(temp_mean)
        multi_lower_bound.append(temp_lower_bound)
        multi_upper_bound.append(temp_upper_bound)

    baseline_mean = []
    baseline_lower_bound = []
    baseline_upper_bound = []

    if baseline is not None:
        for percentage_performance in baseline:
            temp_mean = []
            temp_lower_bound = []
            temp_upper_bound = []
            for margin in margins:
                temp_run_result = []
                for run in range(len(percentage_performance)):
                    inner_run_performance = percentage_performance[run]
                    greater_num = 0
                    for i in inner_run_performance:
                        if i <= margin:
                            greater_num += 1
                    if percentage:
                        temp_run_result.append(greater_num / len(inner_run_performance))
                    else:
                        temp_run_result.append(greater_num)
                success = np.array(temp_run_result)
                success_mean = np.average(success)
                if std:
                    success_var = stats.sem(success)
                else:
                    success_var = np.var(success)
                temp_mean.append(success_mean)
                temp_lower_bound.append(success_mean - success_var)
                temp_upper_bound.append(success_mean + success_var)
            baseline_mean.append(temp_mean)
            baseline_lower_bound.append(temp_lower_bound)
            baseline_upper_bound.append(temp_upper_bound)


    for i in range(len(multi_mean)):
        if subset[i] <= 0.5:
            temp_label = "{}% data threshold".format(subset[i] * 100)
        else:
            split_size = np.gcd(int(subset[i] * 100), 100)
            fold = int(100 / split_size)
            temp_label = "{}-fold threshold".format(fold)
        if graph:
            plt.plot(margins, multi_mean[i], label=temp_label, color=color_array[i])
            plt.fill_between(margins, multi_lower_bound[i], multi_upper_bound[i], alpha=.3, color=color_array[i])

    if baseline is not None:
        for i in range(len(baseline_mean)):
            if subset[i] <= 0.5:
                temp_label = "{}% data lookup".format(subset[i] * 100)
            else:
                split_size = np.gcd(int(subset[i] * 100), 100)
                fold = int(100 / split_size)
                temp_label = "{}-fold lookup".format(fold)
            if graph:
                plt.plot(margins, baseline_mean[i], label=temp_label, color=color_array[i], linestyle='dashed')
                plt.fill_between(margins, baseline_lower_bound[i], baseline_upper_bound[i], alpha=.3, color=color_array[i])
    if graph:
        if vertical_point is not None:
            plt.axvline(x=vertical_point, linestyle='dashed', color="k")
        plt.legend()
        if log:
            plt.xscale('log')
        plt.xlabel("Accuracy")
        if percentage:
            plt.ylabel("Test Success Rate")
        else:
            plt.ylabel("Test Success Sum")
        plt.show()

    return multi_mean, multi_upper_bound, multi_lower_bound, baseline_mean, baseline_upper_bound, baseline_lower_bound

def plot_multiple_accuracy_with_confidence_cross_fold(accuracy, epochs, check_every, subset, std=True,
                                                      first_eval = None, graph=True, color=None):
    if color is None:
        color_array = []

        for i in range(len(accuracy)):
            color_array.append(np.random.rand(3, ))
    else:
        color_array = color


    step = epochs // check_every

    if first_eval is not None:
        eva_epochs = [i * check_every for i in range(step + 1)]
        eva_epochs[0] = first_eval
    else:
        eva_epochs = [(i+1) * check_every for i in range(step)]

    multi_accuracy = []
    multi_accuracy_lower_bound = []
    multi_accuracy_upper_bound = []

    for percentage_performance in accuracy:
        temp_performance_mean = np.average(percentage_performance, axis=0)
        if std:
            temp_performance_var = stats.sem(percentage_performance, axis=0)
        else:
            temp_performance_var = np.var(percentage_performance, axis=0)

        multi_accuracy.append(temp_performance_mean)
        multi_accuracy_lower_bound.append(temp_performance_mean - temp_performance_var)
        multi_accuracy_upper_bound.append(temp_performance_mean + temp_performance_var)

    if graph:
        fig = plt.figure()
        ax = fig.add_subplot()

    for i in range(len(multi_accuracy)):
        if subset[i] <= 0.5:
            temp_label = "{}% data".format(subset[i] * 100)
        else:
            split_size = np.gcd(int(subset[i] * 100), 100)
            fold = int(100 / split_size)
            temp_label = "{}-fold".format(fold)
        if graph:
            ax.plot(eva_epochs, multi_accuracy[i], label=temp_label, color=color_array[i])
            ax.fill_between(eva_epochs, multi_accuracy_lower_bound[i],
                            multi_accuracy_upper_bound[i], alpha=.3, color=color_array[i])

    if graph:
        ax.set_xlim([0, None])
        ax.set_ylim([0, None])
        ax.legend()
        plt.ylabel("Test Success Rate")
        plt.xlabel("Epochs")
        plt.show()

    return multi_accuracy, multi_accuracy_lower_bound, multi_accuracy_upper_bound


def plot_multiple_loss_with_confidence_cross_fold(loss, epochs, subset,loss_name, std=True, graph=True, color=None):
    multi_loss = []
    multi_loss_lower_bounds = []
    multi_loss_upper_bounds = []

    if color is None:
        color_array = []

        for i in range(len(loss)):
            color_array.append(np.random.rand(3, ))
    else:
        color_array = color


    for percentage_loss in loss:
        temp_loss_mean = np.average(percentage_loss, axis=0)
        if std:
            temp_loss_var = stats.sem(percentage_loss, axis=0)
        else:
            temp_loss_var = np.var(percentage_loss, axis=0)

        multi_loss.append(temp_loss_mean)
        multi_loss_lower_bounds.append(temp_loss_mean - temp_loss_var)
        multi_loss_upper_bounds.append(temp_loss_mean + temp_loss_var)

    if graph:
        fig = plt.figure()
        ax = fig.add_subplot()

    for i in range(len(multi_loss)):
        if subset[i] <= 0.5:
            temp_label = "{}% data".format(subset[i] * 100)
        else:
            split_size = np.gcd(int(subset[i] * 100), 100)
            fold = int(100 / split_size)
            temp_label = "{}-fold".format(fold)
        if graph:
            ax.plot(np.arange(epochs), multi_loss[i], label=temp_label, color=color_array[i])
            ax.fill_between(np.arange(epochs), multi_loss_lower_bounds[i], multi_loss_upper_bounds[i], alpha=.3, color=color_array[i])

    if graph:
        ax.set_xlim([0, None])
        ax.set_ylim([0, None])
        ax.legend()
        plt.ylabel("Test {} Loss".format(loss_name))
        plt.xlabel("Epochs")
        plt.show()

    return multi_loss, multi_loss_lower_bounds, multi_loss_upper_bounds