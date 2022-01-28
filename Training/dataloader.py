from torch.utils.data import Dataset
import torch



class CircuitSynthesisGainAndBandwidthManually(Dataset):

    def __init__(self, parameter, result):
        self.parameter = parameter
        self.result = result


    def __len__(self):
        return 0

    def __getitem__(self, index):
        return self.parameter[index], self.result[index]


class MockSimulatorDataset(Dataset):

    def __init__(self, parameters, results):
        self.parameters = parameters
        self.results = results


    def __len__(self):
        return len(parameters)

    def __getitem__(self, index):
        return self.parameters[index], self.results[index]
