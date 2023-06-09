import numpy as np
from dataset import BaseDataset

from utils import generate_metrics_given_config, merge_metrics, run_simulation_given_parameter, \
    generate_performance_diff_metrics, evalCircuit
from model_wrapper import SklearnModelWrapper, PytorchModelWrapper, LookupWrapper

def subset_split(X,y,train_percentage, independent = False):

    split_size = np.gcd(int(train_percentage * 100), 100)
    split_time = int(100 / split_size)
    X_size = X.shape[1]
    combine = np.hstack((X, y))
    

    if independent:
        for _ in range(split_time):
            np.random.shuffle(combine)
            split_array = np.array_split(combine, split_time)
            concat_list = [split_array[k] for k in range(len(split_array)) if k != 0]
            if np.gcd(int(train_percentage * 100), 100) + int(train_percentage * 100) == 100:
                train_data = np.concatenate(concat_list)
                validate_data = split_array[0]
            else:
                train_data = split_array[0]
                validate_data = np.concatenate(concat_list)
            yield train_data[:,:X_size], validate_data[:,:X_size], train_data[:,X_size:], validate_data[:,X_size:]

    else:
        np.random.shuffle(combine)
        split_array = np.array_split(combine, split_time)
        for st in range(len(split_array)):
            concat_list = [split_array[k] for k in range(len(split_array)) if k != st]
            if np.gcd(int(train_percentage * 100), 100) + int(train_percentage * 100) == 100:
                train_data = np.concatenate(concat_list)
                validate_data = split_array[st]
            else:
                train_data = split_array[st]
                validate_data = np.concatenate(concat_list)
            yield train_data[:,:X_size], validate_data[:,:X_size], train_data[:,X_size:], validate_data[:,X_size:]

class EvalModel:
    def __init__(self, train_config, model, train_parameter, train_performance, test_parameter, test_performance, simulator, scaler):
        self.train_config = train_config
        self.model = model
        self.train_parameter = train_parameter
        self.train_performance = train_performance
        self.test_parameter = test_parameter
        self.test_performance = test_performance
        self.simulator = simulator
        self.scaler = scaler

    def eval(self):
        
        train_result = self.model.fit(self.train_performance, self.train_parameter, self.test_performance, self.test_parameter, self.scaler)
        if self.train_config["test_margin_accuracy"]:
            parameter_prediction = self.model.predict(self.test_performance)
            inverse_transform_parameter, inverse_transform_performance = BaseDataset.inverse_transform(parameter_prediction, self.test_performance, self.scaler)
            _, mapping_performance_prediction = run_simulation_given_parameter(self.simulator, inverse_transform_parameter, train=False)
            validate_test_result = generate_performance_diff_metrics(mapping_performance_prediction, inverse_transform_performance, self.simulator, train=False)
            train_result.update(validate_test_result)
        if self.train_config["train_margin_accuracy"]:
            parameter_prediction = self.model.predict(self.train_performance)
            inverse_transform_parameter, inverse_transform_performance = BaseDataset.inverse_transform(parameter_prediction, self.train_performance, self.scaler)
            _, mapping_performance_prediction = run_simulation_given_parameter(self.simulator, inverse_transform_parameter, train=False)
            validate_train_result = generate_performance_diff_metrics(mapping_performance_prediction, inverse_transform_performance, self.simulator, train=True)
            train_result.update(validate_train_result)
        self.model.reset()
        return train_result




class ModelEvaluator:
    def __init__(self, parameter, performance, eval_dataset, metric, simulator, train_config, model):

        if np.any(performance == 0):
            raise ValueError("There is 0 in performance before scaling")

        new_parameter, new_performance, data_scaler = eval_dataset.transform_data(parameter, performance)

        self.parameter = new_parameter
        self.performance = new_performance
        self.simulator = simulator
        self.eval_dataset = eval_dataset
        self.metric = metric
        self.train_config = train_config
        self.scaler = data_scaler

        if train_config["model_type"] == 0:
            self.model_wrapper = SklearnModelWrapper(model)
        elif train_config["model_type"] == 1:
            self.model_wrapper = PytorchModelWrapper(model, train_config, simulator)
        else:
            self.model_wrapper = LookupWrapper(self.simulator.sign)

    def eval(self):

        subset = self.train_config["subset"]
        metrics_dict = generate_metrics_given_config(self.train_config)
        
        if self.train_config["check_circuit"]:
            evalCircuit(self.train_config["num_sample_check"], self.simulator, self.scaler, self.train_config["random_sample_scale"])

        for index, percentage in enumerate(subset):
            print("Running with percentage {}".format(percentage))
            if percentage == 1 or percentage > 1:
                raise ValueError("Subset Percentage must smaller than 1")
            if np.gcd(int(percentage * 100), 100) + int(percentage * 100) != 100 \
                    and np.gcd(int(percentage * 100), 100) != int(percentage * 100):
                raise ValueError("Subset Percentage must be divisble")

            subset_metrics_dict = generate_metrics_given_config(self.train_config)
            count = 0
            for (parameter_train, parameter_test, performance_train, performance_test) in subset_split(self.parameter, self.performance, percentage, self.train_config["independent_sample"]):
                count += 1
                print("Run with {} percentage of training data, Run number {}".format(percentage, count))
                kfold_metrics_dict = generate_metrics_given_config(self.train_config)
                if self.train_config["subset_parameter_check"]:
                    for subset_index in range(index + 1):

                        subset_parameter_percentage = subset[subset_index]
                        subset_parameter_size = int(parameter_train.shape[0] * (subset_parameter_percentage/percentage))

                        subsample_index = np.random.choice(np.arange(0, parameter_train.shape[0]),
                                                           size=subset_parameter_size,
                                                           replace=False)
                        subsample_parameter = parameter_train[subsample_index,:]
                        subsample_performance = performance_train[subsample_index, :]
                        subsample_args = {
                            "parameter": subsample_parameter,
                            "performance": subsample_performance

                        }

                        new_train_parameter, new_train_performance,_ = self.eval_dataset.modify_data(parameter_train,
                                                                                                     performance_train, parameter_test,
                                                                                                     performance_test, train=True, extra_args=subsample_args)
                        new_test_parameter, new_test_performance,_ = self.eval_dataset.modify_data(parameter_train, performance_train, parameter_test, performance_test, train=False)

                        result_eval_model = EvalModel(self.train_config, self.model_wrapper,
                                                      new_train_parameter, new_train_performance,
                                                      new_test_parameter, new_test_performance, self.simulator, self.scaler)
                        subset_kfold_metrics_dict = result_eval_model.eval()
                        merge_metrics(kfold_metrics_dict, subset_kfold_metrics_dict)
                else:
                    new_train_parameter, new_train_performance, _ = self.eval_dataset.modify_data(parameter_train,
                                                                                                  performance_train,
                                                                                                  parameter_test,
                                                                                                  performance_test,
                                                                                                  train=True)
                    new_test_parameter, new_test_performance, _ = self.eval_dataset.modify_data(parameter_train,
                                                                                                performance_train,
                                                                                                parameter_test,
                                                                                                performance_test,
                                                                                                train=False)

                    result_eval_model = EvalModel(self.train_config, self.model_wrapper,
                                                  new_train_parameter, new_train_performance,
                                                  new_test_parameter, new_test_performance, self.simulator, self.scaler)
                    subset_kfold_metrics_dict = result_eval_model.eval()
                    merge_metrics(kfold_metrics_dict, subset_kfold_metrics_dict)
                merge_metrics(subset_metrics_dict, kfold_metrics_dict)
            merge_metrics(metrics_dict, subset_metrics_dict)
        return metrics_dict
