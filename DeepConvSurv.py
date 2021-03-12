import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()

class DeepConvSurv(nn.Module):

    def __init__(self):
        super(DeepConvSurv, self).__init__()
        self.conv1 = nn.Conv2d(3,32,7,stride=3)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32,32,5,stride=2)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(32,32,3,stride=2)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.pool2 = nn.MaxPool2d(2)

        self.conv_module = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool1,
            self.conv2,
            nn.ReLU(),
            self.conv3,
            self.pool2
        )

        self.fc1 = nn.Linear(32*9*9, 32)
        self.fc2 = nn.Linear(32,1)

        self.fc_module = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2
        )

        self.params = list(self.parameters())

        if use_cuda:
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self, x):
        out = self.conv_module(x)
        out = out.view(-1, 81*32)
        out = self.fc_module(out)

        return out
