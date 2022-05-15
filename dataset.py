import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as f
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def load_cifar10(batch=128):
    trainval_dataset=datasets.CIFAR10('./data',
                         train=False,
                         download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                               [0.5, 0.5, 0.5],  # RGB 平均
                               [0.5, 0.5, 0.5]   # RGB 標準偏差
                             )
                         ]))
    """
    transforms.Normalize(
       [0.5, 0.5, 0.5],  # RGB 平均
       [0.5, 0.5, 0.5]   # RGB 標準偏差
       )
    """
    n_samples = len(trainval_dataset) # n_samples is 60000
    train_size = int(n_samples * 0.8) # train_size is 48000

    subset1_indices = list(range(0,train_size)) # [0,1,.....47999]
    subset2_indices = list(range(train_size,n_samples)) # [48000,48001,.....59999]

    train_dataset = Subset(trainval_dataset, subset1_indices)
    val_dataset   = Subset(trainval_dataset, subset2_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=True
    )



    test_loader = DataLoader(
        datasets.CIFAR10('./data',
                         train=False,
                         download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 [0.5, 0.5, 0.5],  # RGB 平均
                                 [0.5, 0.5, 0.5]  # RGB 標準偏差
                             )
                         ])),
        batch_size=batch,
        shuffle=True
    )

    #return train_loader,test_loader,val_dataset
    return train_loader
