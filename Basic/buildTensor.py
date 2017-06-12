import torch

t = torch.Tensor(4,6)
print(t)

t = torch.rand(5,5)
print(t)
print(t.size())

a = torch.rand(4,3)
b = torch.rand(4,3)
print(a+b)