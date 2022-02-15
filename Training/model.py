
import torch.nn as nn


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
    
class Model500(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(Model500, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(parameter_count, 200),
            nn.GELU(),
            nn.Linear(200, 300),
            nn.GELU(),
            nn.Linear(300, 500),
            nn.GELU(),
            nn.Linear(500, 500),
            nn.GELU(),
            nn.Linear(500, 300),
            nn.GELU(),
            nn.Linear(300, 200),
            nn.GELU(),
            nn.Linear(200, output_count)
        )

    def forward(self, x):
        return self.network(x)


class deepModel(nn.Module):
    def __init__(self, input_size=2, output_size=2):
        super(deepModel, self).__init__()
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
        
        







