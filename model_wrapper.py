from dataset import BasePytorchModelDataset
from torch.utils.data import DataLoader

class SklearnModelWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, train_X, train_y, test_X, test_y):
        self.model = self.model.fit(train_X, train_y)
        return {}

    def predict(self, X, y):
        predict = self.model.predict(X)
        return predict


class PytorchModelWrapper:
    def __init__(self, model, optimizer, loss, scaler, check_every, first_eval):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.scaler = scaler

    def fit(self, train_X, train_y, test_X, test_y):
        train_dataset = BasePytorchModelDataset(train_X, train_y)
        test_dataset = BasePytorchModelDataset(test_X, test_y)
        train_dataloader = DataLoader(train_dataset, batch_size=100)
        test_dataloader = DataLoader(test_dataset, batch_size=100)

    def predict(self, X, y):
        test_dataset = BasePytorchModelDataset(X, y)
