
import numpy as np
from torch.utils.data import ConcatDataset


def generate_duplicate_data(train, test, thresholds):
    return_train, return_test = train, test

    for threshold in thresholds:
        new_train = train * threshold
        return_train = np.concatenate((return_train, new_train), axis=0)
        return_test = np.concatenate((return_test, return_test), axis=0)

    return return_train, return_test


def baseline_testing(X_train, X_test, thresholds=None):
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.1]

    total = X_test.shape[0]
    correct = [0 for _ in range(len(thresholds))]

    for datapoint in X_test:
        for index, threshold in enumerate(thresholds):
            for train_datapoint in X_train:
                diff = np.abs((datapoint - train_datapoint)) / datapoint
                if np.all(diff <= threshold):
                    correct[index] += 1
                    break

    return [i / total for i in correct]


def generate_new_dataset_maximum_performance(performance, parameter, order, sign, greater=False):
    # parameter x -> performance y using simulator
    # Go through original Dataset D where D consists of pairs of (x,y)
    # For each pair of (x,y)
    # Go through Dataset D again and find out the maximum y' from pair (x',y')
    # that is greater than y in all performance requirement
    # Generate New pair of (x',y) and put them into new dataset

    num_performance = performance.shape[1]

    def cmp_helper(val1, val2, order):

        for x in order:
            if val1[x] != val2[x]:
                return val1[x] > val2[x]
        return True

    new_performance = []
    new_parameter = []

    for i in range(len(performance)):
        temp_performance = performance[i, :]

        new_temp_parameter = None
        for x in range(len(performance)):
            order_temp_performance = (temp_performance * sign)[np.array(order)]
            order_compare_performance = (performance[x, :] * sign)[np.array(order)]

            if greater:
                if np.all(order_compare_performance > order_temp_performance):
                    if new_temp_parameter is None or cmp_helper(order_compare_performance, new_temp_parameter, order):
                        new_temp_parameter = list(order_compare_performance) + list(parameter[x, :])
            else:
                if np.all(order_compare_performance >= order_temp_performance):
                    if new_temp_parameter is None or cmp_helper(order_compare_performance, new_temp_parameter, order):
                        new_temp_parameter = list(order_compare_performance) + list(parameter[x, :])

        if new_temp_parameter is not None:
            new_performance.append(temp_performance)
            new_parameter.append(new_temp_parameter[num_performance:])


    return np.array(new_performance), np.array(new_parameter)

def get_margin_error(y_hat, y, sign=None):
    sign = np.array(sign)
    temp_y_hat = y_hat
    temp_y = y
    if sign is not None:
        temp_y_hat = y_hat * sign
        temp_y = y * sign

    greater = np.array((temp_y_hat <= temp_y), dtype=int)

    a_err = y_hat - y
    err = np.divide(a_err, y, where=y != 0)

    err = err * greater
    max_err = np.max(np.abs(err), axis=1)

    return max_err

def convert_dataset_to_array(dataset):

    x,y = [], []

    for i in range(len(dataset)):
        temp_x, temp_y = dataset[i]
        x.append(temp_x)
        y.append(temp_y)

    return np.array(x), np.array(y)


def getDatafromDataloader(dataloader):
    try:
        return dataloader.dataset.getAll()
    except:
        perform_array, param_array = [], []
        for i in range(len(dataloader.dataset)):
            temp_perform, temp_param = dataloader.dataset[i]
            perform_array.append(temp_perform)
            param_array.append(temp_param)
        perform_array = np.array(perform_array)
        param_array = np.array(param_array)
        return perform_array, param_array



def save_output_data(baseline=None, test_margins=None, train_margins=None, test_loss=None,
                     train_loss=None, test_accuracy=None, train_accuracy=None):
    pass