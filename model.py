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
        # self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.conv3 = nn.Conv2d(32, 32, 5)
        self.conv4 = nn.Conv2d(32, 32, 5)
        self.conv5 = nn.Conv2d(32, 32, 5)
        self.conv6 = nn.Conv2d(32, 32, 3)
        self.fc1 = nn.Linear(32 * 7 * 12, 4096)
        self.fc2 = nn.Linear(4096, S1 * S2 * T)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x), 0.1))
        # print('after 1st conv layer: ', x.size())
        x = self.pool(F.leaky_relu(self.conv2(x), 0.1))
        # print('after 2nd conv layer: ', x.size())
        x = self.pool(F.leaky_relu(self.conv3(x), 0.1))
        # print('after 3rd conv layer: ', x.size())
        x = self.pool(F.leaky_relu(self.conv4(x), 0.1))
        # print('after 4th conv layer: ', x.size())
        x = self.pool(F.leaky_relu(self.conv5(x), 0.1))
        # print('after 5th conv layer: ', x.size())
        x = self.pool(F.leaky_relu(self.conv6(x), 0.1))
        print('after 6th conv layer: ', x.size())
        x = x.view(-1, 32 * 7 * 12)
        # print('after view: ', x.size())
        # print('before fc1: ', x.size())
        x = F.leaky_relu(self.fc1(x), 0.1)
        # print('after fc1: ', x.size())
        x = self.fc2(x)
        # print('after fc2: ', x.size())
        x = x.view(-1, S1, S2, T)
        return x
