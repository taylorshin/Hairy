import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *

class Hairy(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 7)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 5)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 5)
        # self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 5)
        # self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, 5)
        # self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, 3)
        # YOLO V2
        # self.conv6 = nn.Conv2d(32, T, 3)
        # self.bn6 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 7 * 12, 4096)
        self.fc2 = nn.Linear(4096, S1 * S2 * T)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x), 0.1))
        x = self.pool(F.leaky_relu(self.conv2(x), 0.1))
        x = self.pool(F.leaky_relu(self.conv3(x), 0.1))
        x = self.pool(F.leaky_relu(self.conv4(x), 0.1))
        x = self.pool(F.leaky_relu(self.conv5(x), 0.1))
        x = self.pool(F.leaky_relu(self.conv6(x), 0.1))
        # YOLO v2
        # x = self.pool(self.conv6(x))
        x = x.view(-1, 32 * 7 * 12)
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.fc2(x)
        x = x.view(-1, S1, S2, T)
        return x
