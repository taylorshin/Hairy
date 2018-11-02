import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *

class Hairy(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.conv2 = nn.Conv2d(5, 9, 5)
        self.conv3 = nn.Conv2d(9, 13, 5)
        self.conv4 = nn.Conv2d(13, 17, 5)
        self.conv5 = nn.Conv2d(17, 21, 5)
        self.conv6 = nn.Conv2d(21, 25, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # print('after 1st conv layer (after maxpool): ', x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # print('after 2nd conv layer: ', x.size())
        x = self.pool(F.relu(self.conv3(x)))
        # print('after 3rd conv layer: ', x.size())
        x = self.pool(F.relu(self.conv4(x)))
        # print('after 4th conv layer: ', x.size())
        x = self.pool(F.relu(self.conv5(x)))
        # print('after 5th conv layer: ', x.size())
        x = self.pool(F.relu(self.conv6(x)))
        # print('after 6th conv layer: ', x.size())
        # TODO: fix this hacky solution
        x = x[:, :, :, :-1]
        # print('after hax: ', x.size())
        # x = x.view(-1, 16 * 5 * 5)
        return x
