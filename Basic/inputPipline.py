
import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

train_dataset = dsets.CIFAR10(root='../data/',
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)