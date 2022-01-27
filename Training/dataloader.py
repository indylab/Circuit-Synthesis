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



