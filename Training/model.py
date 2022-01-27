
import torch
import torch.nn as nn


class CSGainAndBandwidthManually(nn.Module):
    def __init__(self, parameter_count = 2, output_count = 2):
        super(CSGainAndBandwidthManually, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 10),
            nn.Linear(10, 30),
            nn.Linear(30, output_count)
        )

    def forward(self, x):
        return self.network(x)




if __name__ == '__main__':
    test = torch.rand(10, 2)
    test_model = CSGainAndBandwidthManually()

    test_out = test_model(test)

    print(test_out.shape)