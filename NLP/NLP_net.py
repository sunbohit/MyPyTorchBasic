import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

linear = nn.Linear(5, 3)
data = autograd.Variable( torch.randn(2, 5) )
print(linear(data))

data = autograd.Variable( torch.randn(2, 2) )
print(data)
print(F.relu(data))

data = autograd.Variable( torch.randn(5) )
print(data)
print(F.softmax(data))
print(F.softmax(data).sum())
print(F.log_softmax(data))

