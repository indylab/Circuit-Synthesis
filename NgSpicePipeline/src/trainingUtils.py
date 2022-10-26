import numpy as np


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


def generate_new_dataset_maximum_performance(performance, parameter, order, sign, duplication, greater=False):
    # parameter x -> performance y using simulator
    # Go through original Dataset D where D consists of pairs of (x,y)
    # For each pair of (x,y)
    # Go through Dataset D again and find out the maximum y' from pair (x',y')
    # that is greater than y in all performance requirement
    # Generate New pair of (x',y) and put them into new dataset

    num_performance = performance.shape[1]

    def cmp_helper(val1, val2):
        # order will be grabbed from parent function argument


        cmp_val1 = val1[:num_performance]
        cmp_val2 = val2[:num_performance]

        cmp_val1 = np.array(cmp_val1) * np.array(sign)
        cmp_val2 = np.array(cmp_val2) * np.array(sign)

        for x in order:
            if cmp_val1[x] != cmp_val2[x]:
                return cmp_val1[x] > cmp_val2[x]

        return True

    new_performance = []
    new_parameter = []

    for i in range(len(performance)):

        temp_performance = performance[i, :]
        temp_new_training_list = []
        best_temp_sample = None

        for x in range(len(performance)):
            order_temp_performance = (temp_performance * sign)[np.array(order)]
            order_compare_performance = (performance[x, :] * sign)[np.array(order)]

            if (greater and np.all(order_compare_performance > order_temp_performance)) or (not greater and np.all(order_compare_performance >= order_temp_performance)):
                new_temp_training_val = list(order_compare_performance) + list(parameter[x, :])
                temp_new_training_list.append(new_temp_training_val)
                if best_temp_sample is None or cmp_helper(order_compare_performance, best_temp_sample):
                    best_temp_sample = new_temp_training_val



        if best_temp_sample is not None:
            if duplication == 0:
                new_performance.append(temp_performance)
                new_parameter.append(best_temp_sample[num_performance:])
            else:
                sorted_new_sample = np.array(sort_nested_list_helper(temp_new_training_list, order, sign)[:duplication+1])

                new_performance = new_performance + [temp_performance for _ in range(len(sorted_new_sample))]
                new_parameter = new_parameter + list(sorted_new_sample[:,num_performance:])


    return np.array(new_performance), np.array(new_parameter)

def sort_nested_list_helper(val_list, order, sign):

    for i in range(len(order)-1, -1, -1):
        val_list = sorted(val_list, key = lambda x: x[order[i]] * sign[i], reverse=True)
    return val_list

def get_margin_error(y_hat, y, sign=None):

    temp_y_hat = y_hat
    temp_y = y
    if sign is not None:
        sign = np.array(sign)
        temp_y_hat = y_hat * sign
        temp_y = y * sign

    greater = np.array((temp_y_hat <= temp_y), dtype=int)

    a_err = y_hat - y

    err = np.divide(a_err, y, where=y != 0)

    err = err * greater

    return np.abs(err)




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



def save_output_data(result_files, circuit_name):

    for k,v in result_files.items():
        save_path = "../result_out/" + circuit_name + "-" + k + ".npy"
        np.save(save_path, v)


def Lourenco_method(param, perform, sign, n=0.15, K = 40):

    new_param = []
    new_perform = []

    for index in range(len(param)):
        new_param.append(param[index])
        new_perform.append(perform[index])
        for k in range(K):
            average_perform = np.average(perform, axis=0)
            random_sample = np.random.rand(*average_perform.shape)
            average_perform = average_perform * sign

            new_param.append(param[index])
            new_perform.append(perform[index] - (n * random_sample) / average_perform)

    return np.array(new_param), np.array(new_perform)


