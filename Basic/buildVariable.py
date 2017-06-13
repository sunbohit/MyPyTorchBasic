import torch
from torch.autograd import Variable

a = Variable(torch.ones(4,4), requires_grad=True)
b = a+1
print(a)
print(b)
print(b.grad_fn)