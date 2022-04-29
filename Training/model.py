import torch.nn as nn


class ResidualNetwork(nn.Module):
    def __init__(self, num_channel):
        super(ResidualNetwork, self).__init__()
        self.linear1 = nn.Linear(num_channel, num_channel)
        self.linear2 = nn.Linear(num_channel, num_channel)
        self.relu = nn.ReLU()

        
    def forward(self, x):
        out1 = self.linear2(self.relu((self.linear1(x))))
        
        return out1 + x
       


class CSGainAndBandwidthManually(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(CSGainAndBandwidthManually, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 30),
            nn.GELU(),
            nn.Linear(30, 60),
            nn.GELU(),
            nn.Linear(60, 120),
            nn.GELU(),
            nn.Linear(120, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 120),
            nn.GELU(),
            nn.Linear(120, 60),
            nn.GELU(),
            nn.Linear(60, 30),
            nn.GELU(),
            nn.Linear(30, output_count)
        )

    def forward(self, x):
        return self.network(x)
    
class Shallow_Net(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(Shallow_Net, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 100),
            nn.Sigmoid(),
            nn.Linear(100, output_count),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)
    
class Sigmoid_Net(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(Sigmoid_Net, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 20),
            nn.Sigmoid(),
            nn.Linear(20, 20),
            nn.Sigmoid(),
            nn.Linear(20, 20),
            nn.Sigmoid(),
            nn.Linear(20, 20),
            nn.Sigmoid(),
            nn.Linear(20, 20),
            nn.Sigmoid(),
            nn.Linear(20, 20),
            nn.Sigmoid(),
            nn.Linear(20, 20),
            nn.Sigmoid(),
            nn.Linear(20, output_count),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)
class WideModel(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(WideModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 200),
            nn.GELU(),
            nn.Linear(200, 2000),
            nn.GELU(),
            nn.Linear(2000, 2000),
            nn.GELU(),
            nn.Linear(2000, 200),
            nn.GELU(),
            nn.Linear(200, output_count)
        )

    def forward(self, x):
        return self.network(x)

class DeepModel(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(DeepModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, output_count)
        )

    def forward(self, x):
        return self.network(x)
    
class Model500GELU(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(Model500GELU, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 200),
            nn.BatchNorm1d(200),
            nn.GELU(),
            nn.Linear(200, 300),
            nn.BatchNorm1d(300),
            nn.GELU(),
            nn.Linear(300, 500),
            nn.BatchNorm1d(500),
            nn.GELU(),
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.GELU(),
            nn.Linear(500, 300),
            nn.BatchNorm1d(300),
            nn.GELU(),
            nn.Linear(300, 200),
            nn.BatchNorm1d(200),
            nn.GELU(),
            nn.Linear(200, output_count)
        )

    def forward(self, x):
        return self.network(x)
    
class Model500RELUBN(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(Model500RELUBN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            ResidualNetwork(500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, output_count)
        )

    def forward(self, x):
        return self.network(x)
    
    
class Model500RELUBNResidual(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(Model500RELUBNResidual, self).__init__()
        self.layer1 = nn.Linear(parameter_count, 200)
        self.layer2 = nn.Linear(200, 500)
        self.layer3 = ResidualNetwork(500)
        self.layer4 = nn.Linear(500, 200)
        self.layer5 = nn.Linear(200, output_count)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(200)
        self.batchnorm2 = nn.BatchNorm1d(500)
        self.batchnorm3 = nn.BatchNorm1d(500)
        self.batchnorm4 = nn.BatchNorm1d(200)


    def forward(self, x):
        out200_1 = self.relu(self.batchnorm1(self.layer1(x)))
        out500_1 = self.relu(self.batchnorm2(self.layer2(out200_1)))
        out500_2 = self.relu(self.batchnorm3(self.layer3(out500_1)))
        out200_2 = self.layer4(out500_2)
        out200_2 = self.relu(self.batchnorm4(out200_1 + out200_2))
        return self.layer5(out200_2)
    
    
class ModelSmallGELU(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(ModelSmallGELU, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 50),
            nn.GELU(),
            nn.Linear(50, 100),
            nn.GELU(),
            nn.Linear(100, 100),
            nn.GELU(),
            ResidualNetwork(100),
            nn.GELU(),
            nn.Linear(100, output_count)
        )

    def forward(self, x):
        return self.network(x)
    
class Model500SiLU(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(Model500SiLU, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 200),
            nn.SiLU(),
            nn.Linear(200, 300),
            nn.SiLU(),
            nn.Linear(300, 500),
            nn.SiLU(),
            nn.Linear(500, 500),
            nn.SiLU(),
            nn.Linear(500, 300),
            nn.SiLU(),
            nn.Linear(300, 200),
            nn.SiLU(),
            nn.Linear(200, output_count)
        )

    def forward(self, x):
        return self.network(x)
    
class Model500Tan(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(Model500Tan, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 200),
            nn.Tanh(),
            nn.Linear(200, 300),
            nn.Tanh(),
            nn.Linear(300, 500),
            nn.Tanh(),
            nn.Linear(500, 500),
            nn.Tanh(),
            nn.Linear(500, 300),
            nn.Tanh(),
            nn.Linear(300, 200),
            nn.Tanh(),
            nn.Linear(200, output_count)
        )

    def forward(self, x):
        return self.network(x)
    
class DeepModel2(nn.Module):
    def __init__(self, input_size=2, output_size=2):
        super(DeepModel2, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 30),
            nn.GELU(),
            nn.Linear(30, 60),
            nn.GELU(),
            nn.Linear(60, 120),
            nn.GELU(),
            nn.Linear(120, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 120),
            nn.GELU(),
            nn.Linear(120, 60),
            nn.GELU(),
            nn.Linear(60, 30),
            nn.GELU(),
            nn.Linear(30, output_size)
        )

    def forward(self, x):
        return self.network(x)
        
        







