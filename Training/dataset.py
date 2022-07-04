from torch.utils.data import Dataset


class CircuitSynthesisGainAndBandwidthManually(Dataset):

    def __init__(self, parameters, results):
        self.parameters = parameters
        self.result = results

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, index):
        return self.parameters[index], self.result[index]

    def getAll(self):
        return self.parameters, self.result


class MockSimulatorDataset(Dataset):

    def __init__(self, parameters, results):
        self.parameters = parameters
        self.results = results

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, index):
        return self.parameters[index], self.results[index]