import torch
from torch.autograd import Variable

a = Variable(torch.ones(4,4), requires_grad=True)
b = a+1
print(a)
print(b)
print(b.creator)

c = b*b*2
d = c.mean()

print(a.grad)

d.backward()
print(a.grad)