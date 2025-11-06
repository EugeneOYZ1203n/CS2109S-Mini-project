import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # input is (3, 32, 32)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1) # output (8, 32, 32)
        self.pool = nn.MaxPool2d(2, 2) # output (12, 16, 16)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # output (8, 16, 16)
        self.fc1 = nn.Linear(16 * 8 * 8, 64) 
        self.fc2 = nn.Linear(64, 17)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



net = Net()