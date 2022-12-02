from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np


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

class ModelEvaluator:
    def __init__(self, x, y, dataset, metric, simulator, train_config, model=RandomForestRegressor()):
        self.x = x
        self.y = y
        self.model = model
        self.simulator = simulator
        self.dataset = dataset
        self.metric = metric
        self.train_config = train_config

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
        for percentage in subset:
            if percentage == 1 or percentage > 1:
                raise ValueError("Subset Percentage must smaller than 1")
            if np.gcd(int(percentage * 100), 100) + int(percentage * 100) != 100 \
                    and np.gcd(int(percentage * 100), 100) != int(percentage * 100):
                raise ValueError("Subset Percentage must be divisble")

            for (X_train, X_test, y_train, y_test) in subset_split(self.x, self.y, percentage):
                pass