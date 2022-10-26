from NgSpiceTraining import *
from torch import optim
from sklearn.preprocessing import MinMaxScaler
from Training.dataset import CircuitSynthesisGainAndBandwidthManually
from trainingUtils import *
import os
from torch.utils.data import random_split, ConcatDataset, DataLoader


def CrossFoldValidationPipeline(simulator, rerun_training, model_template, loss, epochs,
                                check_every, subset, duplication, device='cpu', generate_new_dataset=True, MARGINS=None,
                                selectIndex=None,
                                train_status=False, first_eval=1, random_sample=False, num_sample=None):
    if rerun_training:
        simulator.delete_history_file()
        if random_sample:
            x, y = simulator.run_random_training(num_sample)
        else:
            x, y = simulator.run_training()
    else:
        param_outfile_names = simulator.train_param_filenames  # must be in order
        perform_outfile_names = simulator.train_perform_filenames  # must be in order
        curPath = os.getcwd()
        print(curPath)
        out = os.path.join(curPath, "../out/")

        x, y = simulator.getData(param_outfile_names, perform_outfile_names, out)


    print(device)
    num_param, num_perform = len(simulator.parameter_list), len(simulator.performance_list)

    print(x.shape, y.shape)
    data = np.hstack((x, y))

    scaler_arg = MinMaxScaler(feature_range=(-1, 1))
    scaler_arg.fit(data)
    data = scaler_arg.transform(data)

    param, perform = data[:, :num_param], data[:, num_param:]

    if MARGINS is None:
        MARGINS = [0.01, 0.05, 0.1]
    if selectIndex is None:
        selectIndex = 1

    if generate_new_dataset:

        perform, param = generate_new_dataset_maximum_performance(performance=perform, parameter=param,
                                                                  order=simulator.order, sign=simulator.sign, duplication=duplication)
        print("Leftover Param size")
        print(param.shape)
        print("Leftover Perform Size")
        print(perform.shape)
        print("Unique Param size")
        print(np.unique(param, axis=0).shape)
        print("Unique Perform size")
        print(np.unique(perform, axis=0).shape)


    for i in subset:
        if i == 1 or i > 1:
            raise ValueError
        if np.gcd(int(i * 100), 100) + int(i * 100) != 100 and np.gcd(int(i * 100), 100) != int(i * 100):
            raise ValueError

    baseline, test_margins, train_margins, test_loss, train_loss, test_accuracy, train_accuracy = [], [], [], [], [], [], []
    mean_err = []
    mean_performance_err = []
    mean_baseline_err = []
    mean_baseline_performance_err = []
    mean_err_std = []
    mean_performance_err_std = []
    mean_baseline_err_std = []
    mean_baseline_performance_err_std = []
    for percentage in subset:
        # Find out how many split we have to do
        split_size = np.gcd(int(percentage * 100), 100)
        split_time = int(100 / split_size)
        print("For percentage {}, We split the dataset {} times".format(percentage, split_time))
        Full_dataset = CircuitSynthesisGainAndBandwidthManually(perform, param)

        total_length = len(Full_dataset)
        full_split_len = total_length // split_time
        extra_len = full_split_len + total_length % split_time

        split_len_list = [full_split_len for _ in range(split_time - 1)]
        split_len_list.append(extra_len)

        SplitDataset = random_split(Full_dataset, split_len_list)

        subset_baseline = []
        subset_test_margins = []
        subset_train_margins = []
        subset_test_loss = []
        subset_train_loss = []
        subset_test_accuracy = []
        subset_train_accuracy = []
        subset_baseline_average_err_mean = []
        subset_baseline_average_err_std = []
        subset_baseline_average_err_performance_mean = []
        subset_baseline_average_err_performance_std = []
        subset_err_mean = []
        subset_err_performance_mean = []
        subset_err_std = []
        subset_err_performance_std = []
        for i in range(split_time):

            print('Running with Percentage {} Run Number {}'.format(percentage, i))
            if np.gcd(int(percentage * 100), 100) + int(percentage * 100) == 100:
                concat_list = [SplitDataset[k] for k in range(len(SplitDataset)) if k != i]
                train_dataset = ConcatDataset(concat_list)
                validation_dataset = SplitDataset[i]
            else:
                concat_list = [SplitDataset[k] for k in range(len(SplitDataset)) if k != i]
                train_dataset = SplitDataset[i]
                validation_dataset = ConcatDataset(concat_list)
            model = model_template(num_perform, num_param).to(device)
            optimizer = optim.Adam(model.parameters())

            train_data = DataLoader(train_dataset, batch_size=100)
            val_data = DataLoader(validation_dataset, batch_size=100)

            temp_x_train, temp_y_train = convert_dataset_to_array(train_dataset)
            temp_x_test, temp_y_test = convert_dataset_to_array(validation_dataset)

            baseline_performance_result = generate_baseline_performance(temp_x_train, temp_x_test, simulator.sign)
            subset_baseline_average_err_mean.append(np.average(baseline_performance_result))
            subset_baseline_average_err_std.append(stats.sem(baseline_performance_result))
            subset_baseline_average_err_performance_mean.append(np.average(baseline_performance_result, axis=0))
            subset_baseline_average_err_performance_std.append(stats.sem(baseline_performance_result, axis=0))
            subset_baseline.append(np.max(np.abs(baseline_performance_result), axis=1))

            train_losses, val_losses, train_accs, val_accs, test_margin, train_margin, test_margin_average, \
            test_margin_performance_average, test_margin_std, test_margin_performance_std = train(model, train_data,
                                                                                                  val_data, optimizer,
                                                                                                  loss, scaler_arg,
                                                                                                  simulator,
                                                                                                  first_eval=first_eval,
                                                                                                  device=device,
                                                                                                  num_epochs=epochs,
                                                                                                  margin=MARGINS,
                                                                                                  train_acc=train_status,
                                                                                                  sign=simulator.sign,
                                                                                                  print_every=check_every)

            subset_test_margins.append(test_margin)
            subset_train_margins.append(train_margin)
            subset_test_loss.append(val_losses)
            subset_train_loss.append(train_losses)

            subset_err_mean.append(test_margin_average)
            subset_err_std.append(test_margin_std)
            subset_err_performance_mean.append(test_margin_performance_average)
            subset_err_performance_std.append(test_margin_performance_std)

            temp_train_accuracy_list = []
            temp_test_accuracy_list = []

            for i in val_accs:
                temp_test_accuracy_list.append(i[selectIndex])
            if train_status:
                for i in train_accs:
                    temp_train_accuracy_list.append(i[selectIndex])
            subset_test_accuracy.append(temp_test_accuracy_list)
            subset_train_accuracy.append(temp_train_accuracy_list)

            print(test_margin_average)
            print(test_margin_performance_average)

        baseline.append(subset_baseline)

        test_margins.append(subset_test_margins)
        train_margins.append(subset_train_margins)
        test_loss.append(subset_test_loss)
        train_loss.append(subset_train_loss)
        test_accuracy.append(subset_test_accuracy)
        train_accuracy.append(subset_train_accuracy)

        mean_err.append(np.average(np.array(subset_err_mean)))
        mean_err_std.append(np.average(np.array(subset_err_std)))

        mean_baseline_err.append(np.average(np.array(subset_baseline_average_err_mean)))
        mean_baseline_err_std.append(np.average(np.array(subset_baseline_average_err_std)))

        mean_baseline_performance_err.append(np.average(np.array(subset_baseline_average_err_performance_mean), axis=0))
        mean_baseline_performance_err_std.append(
            np.average(np.array(subset_baseline_average_err_performance_std), axis=0))
        mean_performance_err.append(np.average(np.array(subset_err_performance_mean), axis=0))
        mean_performance_err_std.append(np.average(np.array(subset_err_performance_std), axis=0))

    return baseline, test_margins, train_margins, test_loss, train_loss, test_accuracy, train_accuracy, mean_err, \
           mean_performance_err, mean_baseline_err, mean_baseline_performance_err, mean_err_std, \
           mean_performance_err_std, mean_baseline_err_std, mean_baseline_performance_err_std


def Lourenco_Baseline_comparison(simulator, rerun_training, model_template, loss, epochs,
                        check_every, n, K, device='cpu', MARGINS=None,
                        selectIndex=None,
                        train_status=False, first_eval=1, random_sample=False):
    if rerun_training:
        if random_sample:
            x, y = simulator.run_random_training()
        else:
            x, y = simulator.run_training()
    else:
        param_outfile_names = simulator.train_param_filenames  # must be in order
        perform_outfile_names = simulator.train_perform_filenames  # must be in order
        curPath = os.getcwd()
        print(curPath)
        out = os.path.join(curPath, "../out/")

        x, y = simulator.getData(param_outfile_names, perform_outfile_names, out)

    print(device)
    num_param, num_perform = len(simulator.parameter_list), len(simulator.performance_list)

    print(x.shape, y.shape)
    data = np.hstack((x, y))

    scaler_arg = MinMaxScaler(feature_range=(-1, 1))
    scaler_arg.fit(data)
    data = scaler_arg.transform(data)

    param, perform = data[:, :num_param], data[:, num_param:]

    if MARGINS is None:
        MARGINS = [0.01, 0.05, 0.1]
    if selectIndex is None:
        selectIndex = 1


    Full_dataset = CircuitSynthesisGainAndBandwidthManually(perform, param)

    total_length = len(Full_dataset)
    full_split_len = total_length // 10
    extra_len = full_split_len + total_length % 10

    split_len_list = [full_split_len for _ in range(10 - 1)]
    split_len_list.append(extra_len)

    SplitDataset = random_split(Full_dataset, split_len_list)

    test_margins = []
    train_margins = []
    test_loss = []
    train_loss = []
    test_accuracy = []
    train_accuracy = []
    err_mean = []
    err_performance_mean = []
    err_std = []
    err_performance_std = []


    for i in range(10):
        concat_list = [SplitDataset[k] for k in range(len(SplitDataset)) if k != i]
        train_dataset = ConcatDataset(concat_list)
        validation_dataset = SplitDataset[i]

        temp_train_perform, temp_train_param = convert_dataset_to_array(train_dataset)
        temp_test_perform, temp_test_param = convert_dataset_to_array(validation_dataset)

        new_train_perform, new_train_param = Lourenco_method(temp_train_param, temp_train_perform, simulator.sign, n, K)
        new_test_perform, new_test_param = Lourenco_method(temp_test_param, temp_test_perform, simulator.sign, n, K)

        new_train_dataset = CircuitSynthesisGainAndBandwidthManually(new_train_perform, new_train_param)
        new_test_dataset = CircuitSynthesisGainAndBandwidthManually(new_test_perform, new_test_param)

        train_data = DataLoader(new_train_dataset, batch_size=100)
        val_data = DataLoader(new_test_dataset, batch_size=100)

        model = model_template(num_perform, num_param).to(device)
        optimizer = optim.Adam(model.parameters())

        train_losses, val_losses, train_accs, val_accs, test_margin, train_margin, test_margin_average, \
        test_margin_performance_average, test_margin_std, test_margin_performance_std = train(model, train_data,
                                                                                              val_data, optimizer,
                                                                                              loss, scaler_arg,
                                                                                              simulator,
                                                                                              first_eval=first_eval,
                                                                                              device=device,
                                                                                              num_epochs=epochs,
                                                                                              margin=MARGINS,
                                                                                              train_acc=train_status,
                                                                                              sign=simulator.sign,
                                                                                              print_every=check_every)

        train_loss.append(train_losses)
        test_loss.append(val_losses)
        train_margins.append(train_margin)
        test_margins.append(test_margin)
        temp_train_accuracy_list = []
        temp_test_accuracy_list = []
        for i in val_accs:
            temp_test_accuracy_list.append(i[selectIndex])

        if train_status:
            for i in train_accs:
                temp_train_accuracy_list.append(i[selectIndex])

        test_accuracy.append(temp_test_accuracy_list)
        train_accuracy.append(temp_train_accuracy_list)
        err_mean.append(test_margin_average)
        err_std.append(test_margin_std)
        err_performance_mean.append(test_margin_performance_average)
        err_performance_std.append(test_margin_performance_std)


    final_err_mean = np.average(err_mean)
    final_std_mean = np.average(err_std)
    final_performance_err_mean = np.average(err_performance_mean, axis=0)
    final_performance_std_mean = np.average(err_performance_std, axis=0)

    return test_margins, train_margins, test_loss, train_loss, test_accuracy, train_accuracy, \
           final_err_mean, final_std_mean, final_performance_err_mean, final_performance_std_mean