
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

    # def get_similarity(new, old):
    #     count = 0
    #     for i in range(old.shape[0]):
    #         if np.allclose(old[i, :], new[i, :]): count += 1
    #     print(f"{count}/{old.shape[0]}, {count / old.shape[0]}%. {old.shape[0] - count} diff")

    # print("performance")
    # get_similarity(np.array(new_performance), performance)
    # print("parameter")
    # get_similarity(np.array(new_parameter), parameter)
    return np.array(new_performance), np.array(new_parameter)



if __name__ == '__main__':
    test_perform = np.array([[30, 20], [10, 100], [15, 1], [20, 10]])
    test_parameter = np.array([[10], [30], [50], [20]])
    new_perform, new_parameter = generate_new_dataset_maximum_performance(test_perform, test_parameter, [0, 1], [1, 1])

    print(new_perform)
    print('here')
    print(new_parameter)
