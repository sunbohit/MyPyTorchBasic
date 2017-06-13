import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

V_data = [1.0, 2.0, 3.0]
V = torch.Tensor(V_data)
print(V)

M_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
M = torch.Tensor(M_data)
print(M)

# Create a 3D tensor of size 2x2x2.
T_data = [[[1.,2.], [3.,4.]],
          [[5.,6.], [7.,8.]]]
T = torch.Tensor(T_data)
print(T)

x = torch.randn((3, 4, 5))
print(x)