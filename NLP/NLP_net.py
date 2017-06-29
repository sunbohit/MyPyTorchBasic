import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

linear = nn.Linear(5, 3)
data = autograd.Variable( torch.randn(2, 5) )
print(linear(data))