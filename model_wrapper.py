from dataset import BasePytorchModelDataset, BaseDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import wandb
from utils import run_simulation_given_parameter, generate_performance_diff_metrics
from sklearn.utils.validation import check_is_fitted

class SklearnModelWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, train_X, train_y, test_X, test_y, scaler):
        self.model = self.model.fit(train_X, train_y)
        return {}

    def predict(self, X):
        predict = self.model.predict(X)
        return predict

    def reset(self,):
        print('Reset The model')
        self.model = self.model.__class__()

class LookupWrapper:
    def __init__(self, sign):
        self.train_performance = None
        self.train_parameter = None
        self.sign = sign

    def fit(self, train_X, train_y, test_X, test_y, scaler):
        self.train_performance = train_X
        self.train_parameter = train_y
        return {}

    def predict(self, X):

        sign_performance_train = np.array(self.train_performance) * self.sign
        sign_performance_test = np.array(X) * self.sign

        lookup_parameter_test = []
        for data_index in range(len(sign_performance_test)):
            minimum_err = None
            minimum_index = None
            greater = False
            for cmp_data_index in range(len(sign_performance_train)):
                if np.all(sign_performance_train[cmp_data_index] >= sign_performance_test[data_index]):
                    lookup_parameter_test.append(list(self.train_parameter[cmp_data_index]))
                    greater = True
                    break
                temp_err = (np.abs(sign_performance_train[cmp_data_index] - sign_performance_test[data_index]))
                temp_diff = np.abs(np.divide(temp_err, sign_performance_test[data_index],
                                      where=sign_performance_test[data_index] != 0))

                temp_max_diff = np.max(temp_diff)
                if minimum_err is None or temp_max_diff < minimum_err:
                    minimum_index = cmp_data_index
                    minimum_err = temp_max_diff
            if not greater:
                lookup_parameter_test.append(list(self.train_parameter[minimum_index]))

        lookup_parameter_test = np.array(lookup_parameter_test)

        return lookup_parameter_test


    def reset(self,):
        self.train_performance = None
        self.train_parameter = None


class PytorchModelWrapper:
    def __init__(self, model, train_config, simulator):
        self.model = model
        self.train_config = train_config
        self.simulator = simulator
        self.logging = train_config['log_experiments']
        if self.logging :
            wandb.init(project="circuit_training", config=train_config)
    
    def reset(self,):
        print('Reset The model')
        for layers in self.model.children():
            for layer in layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def fit(self, train_X, train_y, test_X, test_y, scaler):
        train_dataset = BasePytorchModelDataset(train_X, train_y)
        test_dataset = BasePytorchModelDataset(test_X, test_y)
        train_dataloader = DataLoader(train_dataset, batch_size=100)
        test_dataloader = DataLoader(test_dataset, batch_size=100)
        train_result = self.model_train(train_X, test_X, train_dataloader, test_dataloader, scaler)
        return train_result


    def predict(self, X):
        self.model.eval()
        return self.model(torch.Tensor(X).to(self.train_config["device"])).to('cpu').detach().numpy()

    def model_train(self,  train_X, test_X, train_dataloader, test_dataloader, scaler):
        train_loss = nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters())


        train_accs = []
        val_accs = []
        losses = []
        val_losses = []
        device = self.train_config["device"]

        if (self.train_config["test_accuracy_per_epoch"] or self.train_config["train_accuracy_per_epoch"]) and self.train_config["first_eval"] == 0:
            if self.train_config["train_accuracy_per_epoch"]:
                train_accuracy = self.eval_epoch_accuracy(train_X, scaler)
                train_accs.append(train_accuracy)
            if self.train_config["test_accuracy_per_epoch"]:
                test_accuracy = self.eval_epoch_accuracy(test_X, scaler)
                val_accs.append(test_accuracy)

        for epoch in range(self.train_config["epochs"]):
            print('epoch: ', epoch, '')
            self.model.train()
            avg_loss = 0
            val_avg_loss = 0
            for t, (x, y) in enumerate(train_dataloader):
                # Zero your gradient
                optimizer.zero_grad()
                x_var = torch.autograd.Variable(x.type(torch.FloatTensor)).to(device)
                y_var = torch.autograd.Variable(y.type(torch.FloatTensor).float()).to(device)

                scores = self.model(x_var)

                loss = train_loss(scores.float(), y_var.float())

                loss = torch.clamp(loss, max=500000, min=-500000)
                avg_loss += (loss.item() - avg_loss) / (t + 1)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                for t, (x, y) in enumerate(test_dataloader):
                    x_var = x.float().to(device)
                    y_var = y.float().to(device)
                    self.model.eval()
                    scores = self.model(x_var)

                    loss = train_loss(scores.float(), y_var.float())

                    loss = torch.clamp(loss, max=500000, min=-500000)
                    val_avg_loss += (loss.item() - val_avg_loss) / (t + 1)


            losses.append(avg_loss)
            val_losses.append(val_avg_loss)

            if self.logging:
                wandb.log({'train_loss': avg_loss, 'val_loss': val_avg_loss, 'epoch': epoch, })

            if self.train_config["test_accuracy_per_epoch"] or self.train_config["train_accuracy_per_epoch"]:
                if (epoch + 1) == self.train_config["first_eval"] or (epoch + 1) % self.train_config["check_every"] == 0:
                    if self.train_config["train_accuracy_per_epoch"]:
                        train_accuracy = self.eval_epoch_accuracy(train_X, scaler)
                        train_accs.append(train_accuracy)
                        print('train',train_accuracy)

                        if self.logging:
                            wandb.log({'train_accuracy': train_accuracy, 'epoch': epoch,})

                    if self.train_config["test_accuracy_per_epoch"]:
                        test_accuracy = self.eval_epoch_accuracy(test_X, scaler)
                        val_accs.append(test_accuracy)
                        print('test',test_accuracy)

                        if self.logging:
                            wandb.log({ 'test_accuracy': test_accuracy, 'epoch': epoch,})


        result_dict = dict()

        result_dict["train_loss"] = losses
        result_dict["validation_loss"] = val_losses
        if self.train_config["train_accuracy_per_epoch"]:
            result_dict["train_accuracy_per_epoch"] = train_accs
        if self.train_config["test_accuracy_per_epoch"]:
            result_dict["validation_accuracy_per_epoch"] = val_accs

        return result_dict

    def eval_epoch_accuracy(self, X, scaler):
        unique_x = np.unique(X, axis=0)

        parameter_preds = self.predict(unique_x)
        inverse_transform_parameter, inverse_transform_performance = BaseDataset.inverse_transform(parameter_preds,
                                                                                                   unique_x,
                                                                                                   scaler)
        _, mapping_performance_prediction = run_simulation_given_parameter(self.simulator, inverse_transform_parameter, train=False)
        validate_result = generate_performance_diff_metrics(mapping_performance_prediction, inverse_transform_performance,
                                                            self.simulator, train=False)
        max_err = validate_result["test_margins"]
        accuracy_threshold = self.train_config["accuracy_per_epoch_threshold"]

        accuracy_boolean = max_err <= accuracy_threshold

        return accuracy_boolean.sum() / len(accuracy_boolean)
