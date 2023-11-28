import torch
from torch import nn

class EvaluationModel(nn.Module):

    def __init__(self):
        super(EvaluationModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(12, 128, 5, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.linear = nn.Sequential(
            nn.Linear(2048+5, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, meta):
        x = self.conv(x)
        #print(x.shape, meta.shape)
        x = torch.cat((x, meta), 1)
        x = self.linear(x)
        return x