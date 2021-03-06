from __future__ import print_function


from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import hyptorch.nn as hypnn

from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.distances import CosineSimilarity


class Net(nn.Module):
    def __init__(self, dim=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 10, 1)
        self.conv2 = nn.Conv2d(50, 50, 10, 1)
        self.fc1 = nn.Linear(1 * 1 * 50, 50)
        self.fc2 = nn.Linear(50, dim)
        self.tp = hypnn.ToPoincare(
            c=1, train_x=False, train_c=False, ball_dim=dim
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 1* 1 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.tp(x)
        return x
