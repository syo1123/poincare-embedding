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
from dataset import load_cifar10, load_svhn
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


train_loader, svhn_dataset=load_svhn()
model=Net(dim=2)
load_path = 'poincare-embedding_100dim.pt'
load_weights = torch.load(load_path, map_location={'cuda:0': 'cpu'})
model.load_state_dict(load_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
ind=[]
ood=[]
for images, labels in tqdm(val_dataset):
    embeddings = model(images).to('cpu').detach().numpy().copy()
    ind.append(np.linalg.norm(embeddings))

for i, (images, labels) in tqdm(enumerate(svhn_dataset), total=10000):
    embeddings=model(images).to('cpu').detach().numpy().copy()
    ood.append(np.linalg.norm(embeddings))
    if i==10000:break

print(np.mean(np.stack(ind)))
print(np.mean(np.stack(ood)))

plt.hist(ind,bins=100, alpha=0.7)
plt.hist(ood,bins=100, alpha=0.7)
plt.show()
