import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Dummy():
    def __init__(self, order=None, sign=None, debug=False) -> None:
        self.order = order
        self.sign = np.array(sign)
        self.debug = debug

    def preproc_y_prime(self, y_prime):
        epsilon = np.random.uniform(0, self.epsilon)
        if self.debug == True:
            epsilon = 0
        return y_prime * (1 - epsilon * self.sign[self.order])

    def transform_data(self, x, y):
        """
        Preprocess data to be used in the model
        """
        data = np.hstack((x, y))
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(data)
        data = scaler.transform(data)
        self.scaler = scaler
        return data[:, :x.shape[1]], data[:, x.shape[1]:], scaler

    def inverse_transform(self, x, y):
        """
        Inverse transform the data to the original scale
        """
        data = np.hstack((x, y))
        data = self.scaler.inverse_transform(data)
        return data[:, :x.shape[1]], data[:, x.shape[1]:]

    def sort_vectors(self, vectors, candidates_vectors_x):
        """
        Sorts the vectors according to the best perfomance from left to right
        """
        vec = np.copy(vectors)  # perfomance
        x_vecs = np.copy(candidates_vectors_x)  # params
        for idx_axis in range(
                vectors.shape[1]):  # [[perf_11 perf_12 perf_13] [perf_21 perf_22 perf_23] [perf_31 perf_32 perf_33]]
            res = np.argwhere(vec[:, idx_axis] == np.amax(vec[:, idx_axis])).reshape(-1)
            vec = vec[res]
            x_vecs = x_vecs[res]
            if len(res) == 1:
                return x_vecs.reshape(-1), vec.reshape(-1)

    def fit(self, X, y):
        # make permutation of y according to order and sign

        y = y * self.sign
        y = y[:, self.order]
        self.X = X
        self.Y = y
        return self.prepare_partial()

    def inverse_fit(self, y):
        y = y[:, self.order]
        y = y * self.sign
        return y

    def find_max(self, y):
        # find x' =  argmax (y) from pairs of (x,y)
        # Generate New pair of (x',y) and put them into new dataset
        candidates_vectors_x, candidates_vectors_y = self.find_feasible(y)
        x, y_perfomance_max = self.sort_vectors(candidates_vectors_y,
                                                candidates_vectors_x)  # sort_vectors(canditates_vectors_y) -> y
        return x, y_perfomance_max

    def find_feasible(self, y):
        comparison_matrix = y <= self.Y  # [True, False, True]
        comparison_matrix = comparison_matrix.all(axis=1)  # [True True True ] - > True
        candidates_vectors_y = self.Y[comparison_matrix]
        candidates_vectors_x = self.X[comparison_matrix]
        return candidates_vectors_x, candidates_vectors_y


class ArgMaxDataset(Dummy):
    def __init__(self, order, sign, debug=False) -> None:
        super().__init__(order, sign, debug)

    def prepare_partial(self, ):
        new_x = []
        if self.debug:
            debug_x = []
            debug_y = []
        for y in self.Y:
            x_prime, y_prime = self.find_max(y, )
            new_x.append(x_prime)
            if self.debug:
                debug_x.append(x_prime)
                debug_y.append(y_prime)
        if self.debug:
            return np.array(debug_x), np.array(debug_y)
        return np.array(new_x), np.array(self.Y)


class SoftArgMaxDataset(ArgMaxDataset):
    def __init__(self, order, sign, epsilon=0.2, debug=False) -> None:
        super().__init__(order, sign, debug)
        self.epsilon = epsilon

    def prepare_partial(self, ):
        new_x = []
        new_y = []
        for y in self.Y:
            x_prime, y_prime = self.find_max(y, )  # (x',y') = argmax(y)
            y = self.preproc_y_prime(y)  # y*(1-eps*sign)
            new_x.append(x_prime)
            new_y.append(y)

        return np.array(new_x), np.array(new_y)


class LorencoDataset(SoftArgMaxDataset):
    def __init__(self, order, sign, n,k,epsilon=0.2, debug=False) -> None:
        super().__init__(order, sign, debug)
        self.epsilon = epsilon
        self.n = n
        self.k = k

    def prepare_partial(self):
        new_x = []
        new_y = []
        for (x, y) in zip(self.X, self.Y):
            y = self.preproc_y_prime(y)
            new_x.append(x)
            new_y.append(y)

        return np.array(new_x), np.array(new_y)


class HybridDataset(ArgMaxDataset):
    def __init__(self, order, sign, epsilon=0.2, debug=False) -> None:
        super().__init__(order, sign, debug)
        self.epsilon = epsilon

    def prepare_data(self, ):
        new_x = []
        new_y = []
        for y in self.Y:
            x_prime, y_prime = self.find_feasible(y)  # find all feasible (x,y') for y with y' >= y
            y_prime = self.preproc_y_prime(y_prime)  # y*(1-epsilon)

            new_x.extend(x_prime)
            new_y.extend(y_prime)

        return np.array(new_x), np.array(new_y)