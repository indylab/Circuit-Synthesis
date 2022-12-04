import numpy as np
from metrics import get_margin_error
from scipy import stats
from utils import generate_metrics_given_config, merge_metrics

def subset_split(X,y,train_percentage):
    split_size = np.gcd(int(train_percentage * 100), 100)
    split_time = int(100 / split_size)
    X_size = X.shape[1]
    combine = np.hstack((X, y))
    np.random.shuffle(combine)

    split_array = np.array_split(combine, split_time)
    for split_time in range(len(split_array)):
        concat_list = [split_array[k] for k in range(len(split_array)) if k != split_time]
        if np.gcd(int(train_percentage * 100), 100) + int(train_percentage * 100) == 100:
            train_data = np.concatenate(concat_list)
            validate_data = split_array[split_time]
        else:
            train_data = split_array[split_time]
            validate_data = np.concatenate(concat_list)
        yield train_data[:,:X_size], validate_data[:,:X_size], train_data[:,X_size:], validate_data[:,X_size:]

class EvalModel:
    def __init__(self, train_config, model, train_parameter, train_performance, test_parameter, test_performance, simulator):
        self.train_config = train_config
        self.model = model
        self.train_parameter = train_parameter
        self.train_performance = train_performance
        self.test_parameter = test_parameter
        self.test_performance = test_performance
        self.simulator = simulator

    def eval(self):
        train_result = self.model.fit(self.train_performance, self.train_parameter)
        parameter_prediction = self.model.predict(self.test_performance)
        _, mapping_performance_prediction = self.simulator.runSimulation(parameter_prediction, train=False)
        validate_result = self.generate_performance_diff_metrics(mapping_performance_prediction)
        validate_result.update(train_result)

        return validate_result

    def generate_performance_diff_metrics(self, performance_prediction):
        margin_error = get_margin_error(performance_prediction, self.test_performance, self.simulator.sign)
        metrics_dict = dict()
        metrics_dict["circuit_error_average"] = np.average(margin_error)
        metrics_dict["performance_error_average"] = np.average(margin_error, axis=0)
        metrics_dict["circuit_error_std"] = stats.sem(margin_error)
        metrics_dict["performance_error_std"] = stats.sem(margin_error, axis=1)
        metrics_dict["circuit_max_error"] = np.max(margin_error, axis=1)

        return metrics_dict


class ModelEvaluator:
    def __init__(self, parameter, performance, dataset, metric, simulator, train_config, model):
        new_parameter, new_performance, data_scaler = dataset.transform_data(parameter, performance)
        self.parameter = new_parameter
        self.performance = new_performance
        self.model = model
        self.simulator = simulator
        self.dataset = dataset
        self.metric = metric
        self.train_config = train_config
        self.scaler = data_scaler
        self.model_wrapper = None

    def get_prediction(self, data):
        return self.model.predict(data)

    def inverse_data(self, params_predicted, perfomance):
        """reversing back to the original scale"""

        # params_predicted, perfomance = self.dataset.inverse_transform(params_predicted,perfomance) #inverse of scaling
        perfomance = self.dataset.inverse_fit(perfomance)  # change order and sign back
        return params_predicted, perfomance

    def predict(self, params):
        """Predict performance of params"""
        params_scaled = self.dataset.transform_params(params)
        perfomance = self.model.predict(params_scaled)
        perfomance = self.dataset.inverse_fit(perfomance)
        return perfomance


    def eval(self):

        subset = self.train_config["subset"]
        metrics_dict = generate_metrics_given_config(self.train_config)
        for percentage in subset:
            if percentage == 1 or percentage > 1:
                raise ValueError("Subset Percentage must smaller than 1")
            if np.gcd(int(percentage * 100), 100) + int(percentage * 100) != 100 \
                    and np.gcd(int(percentage * 100), 100) != int(percentage * 100):
                raise ValueError("Subset Percentage must be divisble")
            subset_metrics_dict = generate_metrics_given_config(self.train_config)
            for (parameter_train, parameter_test, performance_train, performance_test) in subset_split(self.parameter, self.performance, percentage):

                new_train_parameter, new_train_performance = self.dataset.modify_data(parameter_train, performance_train, train=True)
                new_test_parameter, new_test_performance = self.dataset.modify_data(parameter_test,
                                                                                      performance_test, train=False)

                result_eval_model = EvalModel(self.train_config, self.model_wrapper,
                                              new_train_parameter, new_train_performance, new_test_parameter, new_test_performance, self.simulator)
                kfold_metrics_dict = result_eval_model.eval()
                merge_metrics(subset_metrics_dict, kfold_metrics_dict)
            merge_metrics(metrics_dict, subset_metrics_dict)
        return metrics_dict