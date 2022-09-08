from NgSpiceTraining import *
from torch import optim, cuda
from sklearn.preprocessing import MinMaxScaler
from Training.dataset import CircuitSynthesisGainAndBandwidthManually
from visualutils import *
from trainingUtils import *



def TrainPipeline(simulator, rerun_training, model_template, loss, epochs, runtime = 1,
                  device='cpu', plot_loss=True, generate_new_dataset=True, resplit_dataset=True):
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
    data = np.hstack((x, y))
    scaler_arg = MinMaxScaler()
    scaler_arg.fit(data)
    data = scaler_arg.transform(data)
    param, perform = data[:, :num_param], data[:, num_param:]


    MARGINS = [0.01, 0.05, 0.1]



    # create new D' dataset. Definition in generate_new_dataset_maximum_performance
    if generate_new_dataset:
        perform, param = generate_new_dataset_maximum_performance(performance=perform, parameter=param, order=simulator.order,
                                                                  sign=simulator.sign)
    if not resplit_dataset:
        X_train, X_test, y_train, y_test = train_test_split(perform, param, test_size=0.1)

        train_dataset = CircuitSynthesisGainAndBandwidthManually(X_train, y_train)
        val_dataset = CircuitSynthesisGainAndBandwidthManually(X_test, y_test)

        train_data = DataLoader(train_dataset, batch_size=100)
        val_data = DataLoader(val_dataset, batch_size=100)

    test_margins, train_margins = [],[]

    for run in range(runtime):
        if resplit_dataset:
            X_train, X_test, y_train, y_test = train_test_split(perform, param, test_size=0.1)

            train_dataset = CircuitSynthesisGainAndBandwidthManually(X_train, y_train)
            val_dataset = CircuitSynthesisGainAndBandwidthManually(X_test, y_test)

            train_data = DataLoader(train_dataset, batch_size=100)
            val_data = DataLoader(val_dataset, batch_size=100)
        model = model_template(num_perform, num_param).to(device)
        optimizer = optim.Adam(model.parameters())
        train_losses, val_losses, train_accs, val_accs, test_margin, train_margin = train(model, train_data, val_data, optimizer, loss, scaler_arg,
                                                         simulator, device=device, num_epochs=epochs,
                                                           margin=MARGINS, train_acc=False, sign=simulator.sign)
        test_margins.append(test_margin)
        train_margins.append(train_margin)
        if plot_loss:
            _, ax = plt.subplots()
            ax.set_title("Train and Val Losses")
            ax.set_ylabel(f'Loss')
            ax.set_xlabel(f'Epoch')
            ax.plot(range(len(train_losses)), train_losses)
            ax.plot(range(len(val_losses)), val_losses)
            ax.legend(["Train", "Validation"])
            plt.show()

            _, ax = plt.subplots()
            ax.set_title("Validation Success Rate")
            ax.set_ylabel(f'validation Success rate')
            ax.set_xlabel(f'Epoch')
            ax.plot(range(len(val_accs)), val_accs)
            ax.legend(MARGINS)
            plt.show()

            _, ax = plt.subplots()
            ax.set_title("Training Success Rate")
            ax.set_ylabel(f'train Success rate')
            ax.set_xlabel(f'Epoch')
            ax.plot(range(len(train_accs)), train_accs)
            ax.legend(MARGINS)
            plt.show()


    return test_margins, train_margins