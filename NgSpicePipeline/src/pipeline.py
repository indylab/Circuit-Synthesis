from NgSpiceTraining import *
from Training import models
from torch import optim, cuda
import torch
from sklearn.preprocessing import MinMaxScaler
from Simulator import Simulator
from Training.dataset import CircuitSynthesisGainAndBandwidthManually
from visualutils import *
from trainingUtils import *


def TrainPipeline(simulator, rerun_training, model, loss, epochs, device='cpu'):
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
    model = model(num_perform, num_param).to(device)

    optimizer = optim.Adam(model.parameters())

    print(x.shape, y.shape)
    data = np.hstack((x, y))
    scaler_arg = MinMaxScaler()
    scaler_arg.fit(data)
    data = scaler_arg.transform(data)
    param, perform = data[:, :num_param], data[:, num_param:]

    # Experiment Variables
    SIGN = [1, -1, 1]
    ORDER = [0, 2, 1]
    MARGINS = [0.01, 0.05, 0.1]

    assert (len(SIGN) == len(ORDER)), f"SIGN and ORDER should have the same length. Sign: {len(SIGN)} != Order: {len(ORDER)} "
    assert (len(SIGN) == len(simulator.performance_list)), f"SIGN should have length equal to number of performance. Sign: " \
                                                           f"{len(SIGN)} != Num Params: {len(simulator.performance_list)}"
    assert (len(ORDER) == len(simulator.performance_list)), f"ORDER should have length equal to number of performance. " \
                                                          f"Order: {len(ORDER)} != Num Params: {len(simulator.performance_list)} "

    # create new D' dataset. Definition in generate_new_dataset_maximum_performance
    perform, param = generate_new_dataset_maximum_performance(performance=perform, parameter=param, order=ORDER,
                                                              sign=SIGN)
    X_train, X_test, y_train, y_test = train_test_split(perform, param, test_size=0.1)

    train_dataset = CircuitSynthesisGainAndBandwidthManually(X_train, y_train)
    val_dataset = CircuitSynthesisGainAndBandwidthManually(X_test, y_test)

    train_data = DataLoader(train_dataset, batch_size=100)
    val_data = DataLoader(val_dataset, batch_size=100)
    train_losses, val_losses, train_accs, val_accs, test_margin, train_margin = train(model, train_data, val_data, optimizer, loss, scaler_arg,
                                                     simulator, device=device, num_epochs=epochs,
                                                       margin=MARGINS, train_acc=False, sign=SIGN)
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

    return test_margin