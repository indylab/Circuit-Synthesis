from sklearn.model_selection import train_test_split
from NgSpiceTraining import *
from torch import optim
from sklearn.preprocessing import MinMaxScaler
from Training.dataset import CircuitSynthesisGainAndBandwidthManually
from trainingUtils import *
import os
from torch.utils.data import random_split, ConcatDataset


def TrainPipeline(simulator, rerun_training, model_template, loss, epochs, check_every, runtime = 1,
                  device='cpu', generate_new_dataset=True, resplit_dataset=True, subset=None, MARGINS = None, selectIndex = None, train_status=False):
    if rerun_training:
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

    if subset is None:
        subset = [1]

    print(x.shape, y.shape)
    data = np.hstack((x, y))
    scaler_arg = MinMaxScaler()
    scaler_arg.fit(data)
    data = scaler_arg.transform(data)
    param, perform = data[:, :num_param], data[:, num_param:]


    if MARGINS is None:
        MARGINS = [0.01, 0.05, 0.1]
    if selectIndex is None:
        selectIndex = 1

    # create new D' dataset. Definition in generate_new_dataset_maximum_performance
    if generate_new_dataset:
        perform, param = generate_new_dataset_maximum_performance(performance=perform, parameter=param, order=simulator.order,
                                                                  sign=simulator.sign)
    if not resplit_dataset:
        Full_X_train, X_test, Full_y_train, y_test = train_test_split(perform, param, test_size=0.1)

    test_margins, train_margins = [],[]
    test_accuracy, train_accuracy, = [], []
    test_loss, train_loss = [],[]
    baseline = []
    for run in range(runtime):
        temp_test_margins, temp_train_margins = [], []
        temp_test_accuracy, temp_train_accuracy = [], []
        temp_test_loss, temp_train_loss = [],[]
        temp_baseline = []
        if resplit_dataset:
            Full_X_train, X_test, Full_y_train, y_test = train_test_split(perform, param, test_size=0.1)


        for percentage in subset:
            print('Run Number {} With Subset Percentage {}'.format(run, percentage))
            model = model_template(num_perform, num_param).to(device)
            optimizer = optim.Adam(model.parameters())

            X_train,y_train = generate_subset_data(Full_X_train, Full_y_train, percentage)
            temp_baseline.append(generate_baseline_performance(X_train, X_test, simulator.sign))

            train_dataset = CircuitSynthesisGainAndBandwidthManually(X_train, y_train)
            val_dataset = CircuitSynthesisGainAndBandwidthManually(X_test, y_test)
            print('Length of new dataset is {}'.format(len(train_dataset)))
            train_data = DataLoader(train_dataset, batch_size=100)
            val_data = DataLoader(val_dataset, batch_size=100)

            train_losses, val_losses, train_accs, val_accs, test_margin, train_margin = train(model, train_data, val_data, optimizer, loss, scaler_arg,
                                                             simulator, device=device, num_epochs=epochs,
                                                               margin=MARGINS, train_acc=train_status, sign=simulator.sign, print_every=check_every)
            temp_test_margins.append(test_margin)
            temp_train_margins.append(train_margin)

            temp_test_loss.append(val_losses)
            temp_train_loss.append(train_losses)

            temp_train_accuracy_list = []
            temp_test_accuracy_list = []

            for i in val_accs:
                temp_test_accuracy_list.append(i[selectIndex])
            if train_status:
                for i in train_accs:
                    temp_train_accuracy_list.append(i[selectIndex])
            temp_test_accuracy.append(temp_test_accuracy_list)
            temp_train_accuracy.append(temp_train_accuracy_list)

        test_margins.append(temp_test_margins)
        train_margins.append(temp_train_margins)
        test_loss.append(temp_test_loss)
        train_loss.append(temp_train_loss)
        test_accuracy.append(temp_test_accuracy)
        train_accuracy.append(temp_train_accuracy)
        baseline.append(temp_baseline)
    return baseline, test_margins, train_margins, test_loss, train_loss, test_accuracy, train_accuracy





def CrossFoldValidationPipeline(simulator, rerun_training, model_template, loss, epochs,
                                check_every, subset, device='cpu', generate_new_dataset=True, MARGINS = None, selectIndex = None, train_status = False):
    if rerun_training:
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
    data = np.hstack((x,y))

    scaler_arg = MinMaxScaler()
    scaler_arg.fit(data)
    data = scaler_arg.transform(data)

    param, perform = data[:,:num_param], data[:,num_param:]

    if MARGINS is None:
        MARGINS = [0.01, 0.05, 0.1]
    if selectIndex is None:
        selectIndex = 1

    if generate_new_dataset:
        perform, param = generate_new_dataset_maximum_performance(performance=perform, parameter=param,
                                                                  order=simulator.order, sign=simulator.sign)



    for i in subset:
        if i == 1 or i > 1:
            raise ValueError
        if np.gcd(int(i * 100), 100) + int(i * 100) != 100 and np.gcd(int(i * 100), 100) != int(i * 100):
            raise ValueError

    baseline, test_margins, train_margins, test_loss, train_loss, test_accuracy, train_accuracy = [],[],[],[],[],[],[]

    for percentage in subset:
        # Find out how many split we have to do
        split_size = np.gcd(int(percentage * 100), 100)
        split_time = 100 / split_size
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

        for i in range(len(split_time)):
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
            subset_baseline.append(generate_baseline_performance(temp_x_train, temp_x_test, simulator.sign))
            train_losses, val_losses, train_accs, val_accs, test_margin, train_margin = train(model, train_data,
                                                                                              val_data, optimizer,
                                                                                              loss, scaler_arg,
                                                                                              simulator,
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

            temp_train_accuracy_list = []
            temp_test_accuracy_list = []

            for i in val_accs:
                temp_test_accuracy_list.append(i[selectIndex])
            if train_status:
                for i in train_accs:
                    temp_train_accuracy_list.append(i[selectIndex])
            subset_test_accuracy.append(temp_test_accuracy_list)
            subset_train_accuracy.append(temp_train_accuracy_list)
        baseline.append(subset_baseline)
        test_margins.append(subset_test_margins)
        train_margins.append(subset_train_margins)
        test_loss.append(subset_test_loss)
        train_loss.append(subset_train_loss)
        test_accuracy.append(subset_test_accuracy)
        train_accuracy.append(subset_train_accuracy)

    return baseline, test_margins, train_margins, test_loss, train_loss, test_accuracy, train_accuracy