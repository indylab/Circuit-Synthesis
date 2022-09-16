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


def graph_margin(margin_error, margins, percentage = False):
    counts = []
    margin_error = np.array(margin_error)
    for margin in margins:
        if percentage:
            counts.append((margin_error <= margin).sum() / len(margin_error))
        else:
            counts.append((margin_error <= margin).sum())

    sns.lineplot(x = margins, y = counts)


    plt.xlim(0.5, 0)
    plt.show()

def graph_margin_with_confidence(margin_errors, margins, percentage = True, std=True):
    counts = []

    for margin in margins:
        tmp_margin_counts = []

        for margin_err in margin_errors:
            margin_err = np.array(margin_err)
            if percentage:
                tmp_margin_counts.append((margin_err <= margin).sum() / len(margin_err))
            else:
                tmp_margin_counts.append((margin_err <= margin).sum())
        counts.append(tmp_margin_counts)

    counts_mean = np.array([np.average(i) for i in counts])
    if std:
        counts_var = np.array([np.std(i) for i in counts])
    else:
        counts_var = np.array([np.var(i) for i in counts])

    lower_bound = counts_mean - counts_var
    upper_bound = counts_mean + counts_var


    plt.plot(margins, counts_mean, label="Accuacy at different margin")
    plt.fill_between(margins, lower_bound, upper_bound, alpha=.3)

    plt.legend()
    plt.xscale('log')
    plt.xlabel("Margins")
    if percentage:
        plt.ylabel("Success Percentage")
    else:
        plt.ylabel("Success Amount")
    plt.show()

def graph_multiple_margin_with_confidence(margin_errors, margins, subset,  baseline = None, vertical_point = 0.05, percentage = True, std=True, log=True):

    multi_mean = []
    multi_lower_bound = []
    multi_upper_bound = []
    for index in range(len(margin_errors[0])):
        #loop through each subset
        temp_counts = []
        for margin in margins:
            temp_margin_counts = []
            for run in margin_errors:
                margin_err = np.array(run[index])
                if percentage:
                    temp_margin_counts.append((margin_err <= margin).sum() / len(margin_err))

                else:
                    temp_margin_counts.append((margin_err <= margin).sum())
            temp_counts.append(temp_margin_counts)
        count_mean = np.array([np.average(i) for i in temp_counts])
        if std:
            count_var = np.array([stats.sem(i) for i in temp_counts])
        else:
            count_var = np.array([np.var(i) for i in temp_counts])

        lower_bound = count_mean - count_var
        upper_bound = count_mean + count_var
        multi_mean.append(count_mean)
        multi_lower_bound.append(lower_bound)
        multi_upper_bound.append(upper_bound)

    baseline_mean = []
    baseline_lower_bound = []
    baseline_upper_bound = []

    if baseline is not None:
        for index in range(len(baseline[0])):
            # loop through each subset
            temp_counts = []
            for margin in margins:
                temp_margin_counts = []
                for run in baseline:
                    margin_err = np.array(run[index])
                    if percentage:
                        temp_margin_counts.append((margin_err <= margin).sum() / len(margin_err))
                    else:
                        temp_margin_counts.append((margin_err <= margin).sum())
                temp_counts.append(temp_margin_counts)
            count_mean = np.array([np.average(i) for i in temp_counts])
            if std:
                count_var = np.array([stats.sem(i) for i in temp_counts])
            else:
                count_var = np.array([np.var(i) for i in temp_counts])

            lower_bound = count_mean - count_var
            upper_bound = count_mean + count_var
            baseline_mean.append(count_mean)
            baseline_lower_bound.append(lower_bound)
            baseline_upper_bound.append(upper_bound)

    for i in range(len(multi_mean)):
        plt.plot(margins, multi_mean[i], label="{}% of training data".format(subset[i] * 100))
        plt.fill_between(margins, multi_lower_bound[i], multi_upper_bound[i], alpha=.3)

    if baseline is not None:
        for i in range(len(baseline_mean)):
            plt.plot(margins, baseline_mean[i], label="{}% of training data base".format(subset[i] * 100))
            plt.fill_between(margins, baseline_lower_bound[i], baseline_upper_bound[i], alpha=.3)

    if vertical_point is not None:
        plt.axvline(x=vertical_point, linestyle='dashed', color="k")
    plt.legend()
    if log:
        plt.xscale('log')
    plt.xlabel("Accuracy")
    if percentage:
        plt.ylabel("Success Percentage")
    else:
        plt.ylabel("Success Amount")
    plt.show()

def plot_multiple_accuracy_with_confidence(accuracy, epochs, check_every, subset,  std=True, eva_zero = False):

    step = epochs // check_every
    if eva_zero:
        eva_epochs = [i * check_every for i in range(step + 1)]
    else:
        eva_epochs = [(i+1) * check_every for i in range(step)]

    multi_accuracy = []
    multi_accuracy_lower_bounds = []
    multi_accuracy_upper_bounds = []

    for index in range(len(accuracy[0])):
        subset_performance = []
        for run in range(len(accuracy)):
            subset_performance.append(accuracy[run][index])

        subset_performance = np.array(subset_performance)
        subset_mean = np.mean(subset_performance, axis=0)
        if std:
            subset_var = stats.sem(subset_performance, axis=0)
        else:
            subset_var = np.var(subset_performance, axis=0)

        multi_accuracy.append(subset_mean)
        multi_accuracy_lower_bounds.append(subset_mean - subset_var)
        multi_accuracy_upper_bounds.append(subset_mean + subset_var)

    fig = plt.figure()
    ax = fig.add_subplot()

    for i in range(len(multi_accuracy)):
        ax.plot(eva_epochs, multi_accuracy[i], label="{}% training data".format(subset[i] * 100))
        ax.fill_between(eva_epochs, multi_accuracy_lower_bounds[i], multi_accuracy_upper_bounds[i], alpha=.3)

    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    ax.legend()
    plt.ylabel("Test Success Rate")
    plt.xlabel("Epochs")
    plt.show()

def plot_multiple_loss_with_confidence(loss, epochs, subset,loss_name, std=True):
    multi_loss = []
    multi_loss_lower_bounds = []
    multi_loss_upper_bounds = []

    for index in range(len(loss[0])):
        subset_performance = []
        for run in range(len(loss)):
            subset_performance.append(loss[run][index])

        subset_performance = np.array(subset_performance)
        subset_mean = np.mean(subset_performance, axis=0)
        if std:
            subset_var = stats.sem(subset_performance, axis=0)
        else:
            subset_var = np.var(subset_performance, axis=0)

        multi_loss.append(subset_mean)
        multi_loss_lower_bounds.append(subset_mean - subset_var)
        multi_loss_upper_bounds.append(subset_mean + subset_var)

    fig = plt.figure()
    ax = fig.add_subplot()

    for i in range(len(multi_loss)):
        ax.plot(np.arange(epochs), multi_loss[i], label="{}% of training data".format(subset[i] * 100))
        ax.fill_between(np.arange(epochs), multi_loss_lower_bounds[i], multi_loss_upper_bounds[i], alpha=.3)

    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    ax.legend()
    plt.ylabel("Test {} Loss".format(loss_name))
    plt.xlabel("Epochs")
    plt.show()

def plot_multiple_loss_and_accuracy_with_confidence(loss, accuracy, eva_epochs, epochs, subset, std=True):
    multi_loss = []
    multi_loss_lower_bounds = []
    multi_loss_upper_bounds = []
    multi_accuracy = []
    multi_accuracy_lower_bounds = []
    multi_accuracy_upper_bounds = []


    for index in range(len(loss[0])):
        subset_performance = []
        for run in range(len(loss)):
            subset_performance.append(loss[run][index])

        subset_performance = np.array(subset_performance)
        subset_mean = np.mean(subset_performance, axis=0)
        if std:
            subset_var = stats.sem(subset_performance, axis=0)
        else:
            subset_var = np.var(subset_performance, axis=0)

        multi_loss.append(subset_mean)
        multi_loss_lower_bounds.append(subset_mean - subset_var)
        multi_loss_upper_bounds.append(subset_mean + subset_var)

    for index in range(len(accuracy[0])):
        subset_performance = []
        for run in range(len(accuracy)):
            subset_performance.append(accuracy[run][index])

        subset_performance = np.array(subset_performance)
        subset_mean = np.mean(subset_performance, axis=0)
        if std:
            subset_var = stats.sem(subset_performance, axis=0)
        else:
            subset_var = np.var(subset_performance, axis=0)

        multi_accuracy.append(subset_mean)
        multi_accuracy_lower_bounds.append(subset_mean - subset_var)
        multi_accuracy_upper_bounds.append(subset_mean + subset_var)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax2 = ax.twinx()

    for i in range(len(multi_accuracy)):
        accuracy_color = np.random.rand(3, )
        loss_color = np.random.rand(3, )
        ax.plot(eva_epochs, multi_accuracy[i], label="{}% of training data".format(subset[i] * 100), color=accuracy_color)
        ax.fill_between(eva_epochs, multi_accuracy_lower_bounds[i], multi_accuracy_upper_bounds[i], alpha=.3, color=accuracy_color)
        ax2.plot(np.arange(epochs), multi_loss[i], label="{}% of training data".format(subset[i] * 100), color=loss_color)
        ax2.fill_between(np.arange(epochs), multi_loss_lower_bounds[i], multi_loss_upper_bounds[i], alpha=.3, color=loss_color)



    fig.legend()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax2.set_ylabel("Loss")
    plt.show()



def graph_multiple_margin_with_confidence_cross_fold(margin_errors, margins, subset,  baseline = None, vertical_point = 0.05,
                                                     percentage = True, std=True, log=True):


    num_percentage = len(margin_errors)
    color_array = []

    for i in range(num_percentage):
        color_array.append(np.random.rand(3, ))

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
        if subset[i] > 0.5:
            temp_label = "{}% data threshold".format(subset[i] * 100)
        else:
            split_size = np.gcd(int(subset[i] * 100), 100)
            fold = int(100 / split_size)
            temp_label = "{}-fold threshold".format(fold)
        plt.plot(margins, multi_mean[i], label=temp_label, color=color_array[i])
        plt.fill_between(margins, multi_lower_bound[i], multi_upper_bound[i], alpha=.3, color=color_array[i])

    if baseline is not None:
        for i in range(len(baseline_mean)):
            if subset[i] > 0.5:
                temp_label = "{}% data lookup".format(subset[i] * 100)
            else:
                split_size = np.gcd(int(subset[i] * 100), 100)
                fold = int(100 / split_size)
                temp_label = "{}-fold lookup".format(fold)
            plt.plot(margins, baseline_mean[i], label=temp_label, color=color_array[i], linestyle='dashed')
            plt.fill_between(margins, baseline_lower_bound[i], baseline_upper_bound[i], alpha=.3, color=color_array[i])

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

def plot_multiple_accuracy_with_confidence_cross_fold(accuracy, epochs, check_every, subset, std=True, first_eval = None):
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

    fig = plt.figure()
    ax = fig.add_subplot()

    for i in range(len(multi_accuracy)):
        if subset[i] > 0.5:
            temp_label = "{}% data".format(subset[i] * 100)
        else:
            split_size = np.gcd(int(subset[i] * 100), 100)
            fold = int(100 / split_size)
            temp_label = "{}-fold".format(fold)
        ax.plot(eva_epochs, multi_accuracy[i], label=temp_label)
        ax.fill_between(eva_epochs, multi_accuracy_lower_bound[i], multi_accuracy_upper_bound[i], alpha=.3)

    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    ax.legend()
    plt.ylabel("Test Success Rate")
    plt.xlabel("Epochs")
    plt.show()

    return multi_accuracy, multi_accuracy_lower_bound, multi_accuracy_upper_bound


def plot_multiple_loss_with_confidence_cross_fold(loss, epochs, subset,loss_name, std=True):
    multi_loss = []
    multi_loss_lower_bounds = []
    multi_loss_upper_bounds = []

    for percentage_loss in loss:
        temp_loss_mean = np.average(percentage_loss, axis=0)
        if std:
            temp_loss_var = stats.sem(percentage_loss, axis=0)
        else:
            temp_loss_var = np.var(percentage_loss, axis=0)

        multi_loss.append(temp_loss_mean)
        multi_loss_lower_bounds.append(temp_loss_mean - temp_loss_var)
        multi_loss_upper_bounds.append(temp_loss_mean + temp_loss_var)

    fig = plt.figure()
    ax = fig.add_subplot()

    for i in range(len(multi_loss)):
        if subset[i] > 0.5:
            temp_label = "{}% data".format(subset[i] * 100)
        else:
            split_size = np.gcd(int(subset[i] * 100), 100)
            fold = int(100 / split_size)
            temp_label = "{}-fold".format(fold)
        ax.plot(np.arange(epochs), multi_loss[i], label=temp_label)
        ax.fill_between(np.arange(epochs), multi_loss_lower_bounds[i], multi_loss_upper_bounds[i], alpha=.3)

    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    ax.legend()
    plt.ylabel("Test {} Loss".format(loss_name))
    plt.xlabel("Epochs")
    plt.show()

    return multi_loss, multi_loss_lower_bounds, multi_loss_upper_bounds