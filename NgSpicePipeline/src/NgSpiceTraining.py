import torch
from trainingUtils import *
from scipy import stats

def check_acc(y_hat, y, margins=None):
    if margins is None:
        margins = [0.01, 0.05, 0.1]
    a_err = (np.abs(y_hat - y))  # get normalized error
    err = np.abs(np.divide(a_err, y, where=y != 0))
    assert (err.shape == y.shape)

    accs = []
    for m in margins:
        num_correct = 0
        for row in err:
            num_in_row = len(np.where(row < m)[0])  # margin * 100 because
            if num_in_row == len(row):
                num_correct += 1
        num_samples = y.shape[0]
        print(f"{m}% num correct = {num_correct} / {num_samples}")
        accs.append(num_correct / num_samples)

    return accs


def check_minimum_requirement_acc(y_hat, y, sign, margins=None):
    all_margins = []
    if margins is None:
        margins = [0.01, 0.05, 0.1]
    sign = np.array(sign)

    if sign is not None:
        y_hat = y_hat * sign
        y = y * sign

    for margin in margins:
        greater = np.logical_or((y_hat >= y), (np.abs(np.divide(y_hat - y, y, where=y != 0)) <= margin))

        acc_at_margin = np.all(greater, axis=1).sum() / y_hat.shape[0]
        all_margins.append(acc_at_margin)

    return all_margins



def simulate_points(paramater_preds, norm_perform, scaler, simulator, sign, final = False):
    num_param, num_perform = len(simulator.parameter_list), len(simulator.performance_list)
    data = np.hstack((paramater_preds, norm_perform))
    unnorm_param_preds, unnorm_true_perform = scaler.inverse_transform(data)[:, :num_param], scaler.inverse_transform(
        data)[:,num_param:]

    _, y_sim = simulator.runSimulation(unnorm_param_preds)

    if final:

        return get_margin_error(y_sim, unnorm_true_perform, sign)
    else:
        assert y_sim.shape == unnorm_true_perform.shape, f"simulation failed, {y_sim.shape} != {unnorm_true_perform.shape}"
        assert y_sim.shape == norm_perform.shape, f"simulation failed, {y_sim.shape} != {norm_perform.shape}"


        accs = check_minimum_requirement_acc(y_sim, unnorm_true_perform, sign)

        return accs


def sklearn_train(model, train_data, val_data, scaler, simulator, subfeasible, sign=None):


    X_train, y_train = getDatafromDataloader(train_data)
    X_test, y_test = getDatafromDataloader(val_data)

    X_test = generate_subfeasible_data(X_test, subfeasible, sign)

    model.fit(X_train, y_train)
    predict_param = model.predict(X_test)
    test_margin_whole = simulate_points(predict_param, X_test, scaler, simulator, sign, final=True)
    test_margin_average = np.average(test_margin_whole)
    test_margin_performance_average = np.average(test_margin_whole, axis=0)
    test_margin_std = stats.sem(test_margin_whole)
    test_margin_performance_std = stats.sem(test_margin_whole, axis=0)
    test_margin = np.max(test_margin_whole, axis=1)
    return test_margin_average, test_margin_performance_average, test_margin_std, test_margin_performance_std, test_margin


def train(model, train_data, val_data, optimizer, loss_fn, scaler, simulator, subfeasible, first_eval=0, device='cpu', num_epochs=1000,
          margin=None, train_acc=False, sign=None, print_every = 200):
    if margin is None:
        margin = [0.05]

    train_accs = []
    val_accs = []
    losses = []
    val_losses = []

    final_test_param = None
    final_test_perform = None
    final_train_param = None
    final_train_perform = None

    if first_eval is None:
        first_eval = -1

    if first_eval == 0:
        norm_perform, _ = getDatafromDataloader(val_data)
        norm_perform = np.unique(norm_perform, axis=0)
        model.eval()
        norm_perform = generate_subfeasible_data(norm_perform, subfeasible, sign)
        paramater_preds = model(torch.Tensor(norm_perform).to(device)).to('cpu').detach().numpy()
        acc_list = simulate_points(paramater_preds, norm_perform, scaler, simulator, margin, sign)
        val_accs.append(acc_list)
        print(f"Validation Accuracy Before Training")
        if train_acc:
            norm_perform, _ = getDatafromDataloader(train_data)
            norm_perform = np.unique(norm_perform, axis=0)
            model.eval()
            simulator.save_error_log = True
            paramater_preds = model(torch.Tensor(norm_perform)).detach().numpy()
            acc_list = simulate_points(paramater_preds, norm_perform, scaler, simulator, margin, sign)
            train_accs.append(acc_list)
            print(f"Training_Accuracy at Epoch Before Training")


    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0
        val_avg_loss = 0
        for t, (x, y) in enumerate(train_data):
            # Zero your gradient
            optimizer.zero_grad()
            x_var = torch.autograd.Variable(x.type(torch.FloatTensor)).to(device)
            y_var = torch.autograd.Variable(y.type(torch.FloatTensor).float()).to(device)

            scores = model(x_var)

            loss = loss_fn(scores.float(), y_var.float())

            loss = torch.clamp(loss, max=500000, min=-500000)
            avg_loss += (loss.item() - avg_loss) / (t + 1)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for t, (x, y) in enumerate(val_data):

                x_var = x.float().to(device)
                y_var = y.float().to(device)
                model.eval()
                scores = model(x_var)

                loss = loss_fn(scores.float(), y_var.float())

                loss = torch.clamp(loss, max=500000, min=-500000)
                val_avg_loss += (loss.item() - val_avg_loss) / (t + 1)

        losses.append(avg_loss)
        val_losses.append(val_avg_loss)
        print("Validation Loss for {} epoch is {}, Training Loss for {} epoch is {}".format(epoch, val_avg_loss, epoch, avg_loss))

        if (epoch + 1) == first_eval or (epoch + 1) % print_every == 0 or (num_epochs < print_every and epoch == num_epochs - 1):
            print('t = %d, loss = %.4f' % (epoch + 1, avg_loss))
            print('t = %d, val loss = %.4f' % (epoch + 1, val_avg_loss))
            norm_perform, _ = getDatafromDataloader(val_data)
            norm_perform = np.unique(norm_perform, axis=0)

            model.eval()
            norm_perform = generate_subfeasible_data(norm_perform, subfeasible, sign)
            paramater_preds = model(torch.Tensor(norm_perform).to(device)).to('cpu').detach().numpy()
            acc_list = simulate_points(paramater_preds, norm_perform, scaler, simulator, sign, final=False)
            final_test_param = paramater_preds
            final_test_perform = norm_perform
            val_accs.append(acc_list)
            print(f"Validation Accuracy at Epoch {epoch} = {val_accs[-1][1]}")

            if train_acc:
                norm_perform, _ = getDatafromDataloader(train_data)
                norm_perform = np.unique(norm_perform, axis=0)
                model.eval()
                simulator.save_error_log = True

                paramater_preds = model(torch.Tensor(norm_perform)).detach().numpy()
                acc_list = simulate_points(paramater_preds, norm_perform, scaler, simulator, sign, final=False)
                train_accs.append(acc_list)
                print(f"Training_Accuracy at Epoch {epoch} = {train_accs[-1][0]}")
                final_train_param = paramater_preds
                final_train_perform = norm_perform

    test_margin_whole = simulate_points(final_test_param, final_test_perform, scaler, simulator, sign, final=True)
    test_margin_average = np.average(test_margin_whole)
    test_margin_performance_average = np.average(test_margin_whole, axis=0)
    test_margin_std = stats.sem(test_margin_whole)
    test_margin_performance_std = stats.sem(test_margin_whole, axis=0)
    test_margin = np.max(test_margin_whole, axis=1)


    if train_acc:
        train_margin_whole = simulate_points(final_train_param, final_train_perform, scaler, simulator, sign, final=True)
        train_margin = np.max(train_margin_whole, axis=1)
    else:
        train_margin = []
    return losses, val_losses, train_accs, val_accs, test_margin, train_margin, test_margin_average, \
           test_margin_performance_average, test_margin_std, test_margin_performance_std



def generate_subset_data(Train, Test, percentage):

    subset_index = np.random.choice(np.arange(Train.shape[0]), int(percentage * Train.shape[0]), replace=False)

    return Train[subset_index,:], Test[subset_index,:]


def generate_baseline_performance(X_train, X_test, sign):

    unique_X_train = np.unique(X_train, axis=0)
    unique_X_test = np.unique(X_test, axis=0)

    temp_X_train = unique_X_train * sign
    temp_X_test = unique_X_test * sign


    temp_y_hat = []

    for data in range(len(unique_X_test)):
        minimum_err = None
        minimum_index = None
        greater = False
        for cmp_data_index in range(len(unique_X_train)):
            if np.all(temp_X_train[cmp_data_index] >= temp_X_test[data]):
                temp_y_hat.append(list(unique_X_train[cmp_data_index]))
                greater = True
                break
            temp_err = (np.abs(temp_X_train[cmp_data_index] - temp_X_test[data]))
            temp_diff = np.divide(temp_err, temp_X_test[data], where=temp_X_test[data] != 0)

            temp_max_diff = np.max(temp_diff)
            if minimum_err is None or temp_max_diff < minimum_err:
                minimum_index = cmp_data_index
                minimum_err = temp_max_diff
        if not greater:
            temp_y_hat.append(list(unique_X_train[minimum_index]))

    temp_y_hat = np.array(temp_y_hat)

    return get_margin_error(temp_y_hat, unique_X_test, sign)



