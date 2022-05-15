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
from model import Net
from dataset import load_cifar10
from tqdm import tqdm
import torch.optim as optim


criterion=hypnn.TripletLossh()
train_loader=load_cifar10()
model=Net()
num_epoch=30
optimizer = optim.Adam(model.parameters(), lr=0.001)
losses=[]

for epoch in range(num_epoch):
    for images, labels in tqdm(train_loader):
        optimizer.zero_grad()
        embeddings = model(images)
        loss=criterion(embeddings, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss)

    print(torch.mean(torch.stack(losses)))