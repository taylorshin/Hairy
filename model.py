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
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.conv3 = nn.Conv2d(32, 32, 5)
        self.conv4 = nn.Conv2d(32, 32, 5)
        self.conv5 = nn.Conv2d(32, 32, 5)
        self.conv6 = nn.Conv2d(32, 25, 3)

    def forward(self, x):
        x = self.pool((self.conv1(x)))
        # print('after 1st conv layer: ', x.size())
        x = self.pool((self.conv2(x)))
        # print('after 2nd conv layer: ', x.size())
        x = self.pool((self.conv3(x)))
        # print('after 3rd conv layer: ', x.size())
        x = self.pool((self.conv4(x)))
        # print('after 4th conv layer: ', x.size())
        x = self.pool((self.conv5(x)))
        # print('after 5th conv layer: ', x.size())
        x = self.pool(torch.tanh(self.conv6(x)))
        # print('after 6th conv layer: ', x.size())
        return x
