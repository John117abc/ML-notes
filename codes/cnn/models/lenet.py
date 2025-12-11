import torch
from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self,num_classes=10,in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,kernel_size = 5)
        self.conv2 = nn.Conv2d(6,16,kernel_size=5)
        self.fc1 = nn.Linear(16*4*4,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


