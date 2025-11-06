import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, categories):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) 
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1) 
        self.bn3 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 4 * 4, 64) 
        self.fc2 = nn.Linear(64, categories)

        self.dropout = nn.Dropout(0.2)


    def forward(self, x): # 3 x 32 x 32
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 8 x 16 x 16
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 16 x 8 x 8
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # 32 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



