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
    def __init__(self, order,sign, train_config) -> None:
        self.order = order
        self.sign = np.array(sign)
        self.train_config = train_config

    @staticmethod
    def transform_data(parameter, performance):
        """
        Preprocess data to be used in the model by scaling the data to be in the range of [-1, 1]
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

    def modify_data(self, train_parameter, train_performance, test_parameter, test_performance, train=True):
        if train:
            return train_parameter, train_performance
        else:
            return test_parameter, test_performance


class LorencoDataset(BaseDataset):
    def __init__(self, order, sign, n, K, train_config) -> None:
        super().__init__(order,sign, train_config)
        self.n = n
        self.K = K

    def modify_data(self, train_parameter, train_performance, test_parameter, test_performance, train=True):
        if train:
            return self.LourencoMethod(train_parameter, train_performance)
        else:
            return test_parameter, test_performance

    def LourencoMethod(self, parameter, performance):
        new_param = []
        new_perform = []
        average_perform = np.average(performance, axis=0)

        for index in range(len(parameter)):
            new_param.append(parameter[index])
            new_perform.append(performance[index])

        for k in range(self.K):
            rand_delta = np.random.rand(*performance.shape)
            random_sample = (rand_delta * self.n)/ average_perform
            
            for idx_axis in range(len(self.sign)):
                if self.sign[idx_axis] == 1:
                    random_sample[:,idx_axis] -= random_sample[:,idx_axis]
                else:
                    random_sample[:,idx_axis] += random_sample[:, idx_axis]

            new_perform.append(performance - random_sample) 
            new_param.append(parameter)

    
        return np.vstack(new_param), np.vstack(new_perform)


class ArgMaxDataset(BaseDataset):
    def __init__(self, order, sign, train_config) -> None:
        super().__init__(order, sign, train_config)

    def find_best_performance(self, parameter, performance):
        """
        Sorts the vectors according to the best perfomance from left to right
        """
        if parameter.shape[0] == 0:
            return -1, -1, False
        fit_parameter, fit_performance = self.fit(parameter, performance)

        for idx_axis in range(performance.shape[1]):

            res = np.argwhere(fit_performance[:,idx_axis] == np.amax(fit_performance[:,idx_axis])).reshape(-1)

            fit_performance = fit_performance[res]
            fit_parameter = fit_parameter[res]
            if len(res) == 1:
                return fit_parameter.reshape(-1), fit_performance.reshape(-1), True

    def find_max(self, parameter, performance, temp_performance):
        # find x' =  argmax (y) from pairs of (x,y)
        # Generate New pair of (x',y) and put them into new dataset

        candidates_vector_parameter, candidates_vector_performance = self.find_feasible(parameter, performance, temp_performance)
        sort_vector_parameter, sort_vector_performance, find_max_boolean = self.find_best_performance(candidates_vector_parameter,
                                                                           candidates_vector_performance)
        return sort_vector_parameter, sort_vector_performance, find_max_boolean

    def find_feasible(self,parameter, performance, temp_performance):
        # slow ... calls for each point in the dataset 
        _, fit_performance = self.fit(parameter, performance)

        fit_temp_performance = (temp_performance*self.sign)[self.order]

        comparison_matrix = (fit_temp_performance <= fit_performance).all(axis=1)

        return parameter[comparison_matrix], performance[comparison_matrix]


    def modify_data(self, train_parameter, train_performance, test_parameter, test_performance, train=True):
        if train:
            return self.argmaxModifyData(train_parameter, train_performance)
        else:
            if self.train_config["evaluation_same_distribution"]:
                return self.argmaxModifyData(test_parameter, test_performance, train_parameter, train_performance)
            else:
                return test_parameter, test_performance

    def argmaxModifyData(self, parameter, performance, same_dist_parameter = None, same_dist_performance = None):
        new_parameter = []
        argmax_ratio = 0
        for (temp_performance,temp_parameter) in zip(performance,parameter):
            if same_dist_parameter is None:
                new_temp_parameter, _, _ = self.find_max(parameter, performance, temp_performance)

            else:
                new_temp_parameter, _, find_max_boolean = self.find_max(same_dist_parameter, same_dist_performance, temp_performance)
                if not find_max_boolean:
                    new_temp_parameter, _, _ = self.find_max(parameter, performance, temp_performance)
            if (new_temp_parameter != temp_parameter).all():
                argmax_ratio += 1
            new_parameter.append(new_temp_parameter)
        print(f'Argmax ratio is {argmax_ratio/len(parameter)} with argmax replaced {argmax_ratio} times')
        return np.array(parameter),np.array(performance)


class SoftArgMaxDataset(ArgMaxDataset):
    def __init__(self, order, sign, train_config, epsilon=0.0) -> None:
        super().__init__(order, sign, train_config)
        self.epsilon = epsilon
        print(f'Epsilon is {epsilon}')

    def scale_down_data(self, parameter, performance):
        random_scale = np.random.uniform(0, self.epsilon, size=performance.shape)
        absolute_performance = np.absolute(performance)

        scale_down_value = random_scale * absolute_performance
        scale_down_performance = np.copy(performance)
        for idx_axis in range(len(self.sign)):
            if self.sign[idx_axis] == 1:
                scale_down_performance[:,idx_axis] -= scale_down_value[:,idx_axis]
            else:
                scale_down_performance[:,idx_axis] += scale_down_value[:, idx_axis]
        return parameter, scale_down_performance

    def modify_data(self, train_parameter, train_performance, test_parameter, test_performance, train=True):
        if train:
            parameter, scale_down_performance = self.scale_down_data(train_parameter, train_performance)
            return super().argmaxModifyData(parameter, scale_down_performance)
        else:
            return test_parameter, test_performance


class AblationDuplicateDataset(SoftArgMaxDataset):
    def __init__(self, order, sign, duplication, train_config, epsilon=0.2) -> None:
        super().__init__(order, sign, train_config, epsilon)
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

    def modify_data(self, train_parameter, train_performance, test_parameter, test_performance, train=True):
        if train:
            return self.ablationModifyData(train_parameter, train_performance)
        else:
            return test_parameter, test_performance

    def ablationModifyData(self, parameter, performance, train=True):
        scale_down_parameter, scale_down_performance = self.scale_down_data(parameter, performance)

        new_parameter, new_performance = [], []
        for temp_performance in scale_down_performance:
            new_temp_parameters, new_temp_performances = self.generate_duplication_data(parameter, performance, temp_performance, train)
            
            new_parameter += list(new_temp_parameters)
            new_performance += list(new_temp_performances)

        return np.array(new_parameter), np.array(new_performance)