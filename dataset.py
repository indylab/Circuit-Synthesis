import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class BasePytorchModelDataset(Dataset):
    def __init__(self, performance, parameters):
        self.parameters = np.array(parameters)
        self.performance = np.array(performance)

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, index):
        return self.performance[index], self.parameters[index]

    def getAll(self):
        return self.performance, self.parameters


class BaseDataset:
    def __init__(self, order,sign) -> None:
        self.order = order
        self.sign = np.array(sign)

    @staticmethod
    def transform_data(parameter, performance):
        """
        Preprocess data to be used in the model
        """
        data = np.hstack((np.copy(parameter), np.copy(performance)))
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(data)
        scaled_data = scaler.transform(data)
        return scaled_data[:, :parameter.shape[1]], scaled_data[:, parameter.shape[1]:], scaler

    @staticmethod
    def inverse_transform(parameter, performance, scaler):
        """
        Inverse transform the data to the original scale
        """
        data = np.hstack((parameter, performance))
        data = scaler.inverse_transform(data)
        return data[:, :parameter.shape[1]], data[:, parameter.shape[1]:]


    def fit(self, parameter, performance):
        # make permutation of y according to order and sign

        fit_performance = np.copy(performance) * self.sign
        fit_performance = fit_performance[:, self.order]

        return parameter,fit_performance

    def inverse_fit(self, parameter,performance):
        reverse_order = [0 for _ in range(len(self.order))]
        for index in range(len(self.order)):
            reverse_order[self.order[index]] = index
        inverse_fit_performance = np.copy(performance)[:, reverse_order]
        inverse_fit_performance = inverse_fit_performance * self.sign
        return parameter,inverse_fit_performance

    def modify_data(self, parameter, performance, train=True):
        return parameter, performance


class LorencoDataset(BaseDataset):
    def __init__(self, order, sign, n, K) -> None:
        super().__init__(order,sign)
        self.n = n
        self.K = K

    def modify_data(self, parameter, performance, train=True):
        new_param = []
        new_perform = []
        average_perform = np.average(performance, axis=0)
        for index in range(len(parameter)):
            new_param.append(parameter[index])
            new_perform.append(performance[index])
            for k in range(self.K):
                random_sample = np.random.rand(*average_perform.shape)
                average_perform = average_perform

                new_param.append(parameter[index])
                new_perform.append(performance[index] - (self.n * random_sample) / average_perform)

        return np.array(new_param), np.array(new_perform)


class ArgMaxDataset(BaseDataset):
    def __init__(self, order, sign) -> None:
        super().__init__(order, sign)

    def find_best_performance(self, parameter, performance):
        """
        Sorts the vectors according to the best perfomance from left to right
        """

        fit_parameter, fit_performance = self.fit(parameter, performance)

        for idx_axis in range(performance.shape[1]):
            res = np.argwhere(fit_performance[:,idx_axis] == np.amax(fit_performance[:,idx_axis])).reshape(-1)
            fit_performance = fit_performance[res]
            fit_parameter = fit_parameter[res]
            if len(res) == 1:
                return fit_parameter.reshape(-1), fit_performance.reshape(-1)

    def find_max(self, parameter, performance, temp_performance):
        # find x' =  argmax (y) from pairs of (x,y)
        # Generate New pair of (x',y) and put them into new dataset
        candidates_vector_parameter, candidates_vector_performance = self.find_feasible(parameter, performance, temp_performance)
        sort_vector_parameter, sort_vector_performance = self.find_best_performance(candidates_vector_parameter,
                                                                           candidates_vector_performance)
        return sort_vector_parameter, sort_vector_performance

    def find_feasible(self,parameter, performance, temp_performance):

        fit_parameter, fit_performance = self.fit(parameter, performance)

        fit_temp_performance = (temp_performance*self.sign)[self.order]

        comparison_matrix = (fit_temp_performance <= fit_performance).all(axis=1)

        return parameter[comparison_matrix], performance[comparison_matrix]


    def modify_data(self, parameter, performance, train=True):
        new_parameter = []

        for temp_performance in performance:
            new_temp_parameter, _ = self.find_max(parameter, performance, temp_performance)
            new_parameter.append(new_temp_parameter)

        return np.array(new_parameter),np.array(performance)


class SoftArgMaxDataset(ArgMaxDataset):
    def __init__(self, order, sign, epsilon=0.2) -> None:
        super().__init__(order, sign)
        self.epsilon = epsilon

    def scale_down_data(self, parameter, performance):
        random_scale = np.random.uniform(0, self.epsilon, size=performance.shape)
        absolute_performance = np.absolute(performance)

        scale_down_value = random_scale * absolute_performance
        scale_down_performance = np.copy(performance)
        for idx_axis in range(len(self.order)):
            if self.order[idx_axis] == 1:
                scale_down_performance[:,idx_axis] -= scale_down_value[:,idx_axis]
            else:
                scale_down_performance[:,idx_axis] += scale_down_value[:, idx_axis]
        return parameter, scale_down_performance

    def modify_data(self, parameter, performance, train=True):

        parameter, scale_down_performance = self.scale_down_data(parameter, performance)

        return super().modify_data(parameter, scale_down_performance)

class AblationDuplicateDataset(SoftArgMaxDataset):
    def __init__(self, order, sign, duplication, epsilon=0.2) -> None:
        super().__init__(order, sign, epsilon)
        self.duplication = duplication

    def sort_vectors(self, parameter, performance):

        # We stack performance first because of self.order
        data = np.hstack((performance, parameter))

        for i in range(len(self.order) - 1, -1, -1):
            data = sorted(data, key=lambda x: x[self.order[i]], reverse=True)
        data = np.array(data)
        return data[:, performance.shape[1]:], data[:, :performance.shape[1]]

    def generate_duplication_data(self, parameter, performance, temp_performance, train):
        candidates_vector_parameter, candidates_vector_performance = self.find_feasible(parameter, performance,
                                                                                        temp_performance)

        sort_vector_parameter, sort_vector_performance = self.sort_vectors(candidates_vector_parameter, candidates_vector_performance)

        if train:
            num_sample = self.duplication + 1
        else:
            num_sample = 1
        return sort_vector_parameter[:num_sample], sort_vector_performance[:num_sample]

    def modify_data(self, parameter, performance, train=True):
        scale_down_parameter, scale_down_performance = self.scale_down_data(parameter, performance)

        new_parameter, new_performance = [], []
        for temp_performance in scale_down_performance:
            new_temp_parameters, new_temp_performances = self.generate_duplication_data(parameter, performance, temp_performance, train)
            new_parameter += new_temp_parameters
            new_performance += new_temp_performances

        return np.array(new_parameter), np.array(new_performance)


