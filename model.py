import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *

class Hairy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 7, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        print('input shape: ', x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        print('after 1st conv layer: ', x.shape)
        # x = self.pool(F.relu(self.conv2(x)))
        # print('after 2nd conv layer: ', x.shape)
        x = x.view(-1, 16 * 5 * 5)
        return x
