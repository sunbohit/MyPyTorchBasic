import torch

t = torch.Tensor(4,6)
print(t)

t = torch.rand(5,5)
print(t)
print(t.size())

a = torch.rand(4,3)
b = torch.rand(4,3)
print(a)
print(b)
print('---------')
print(a+b)
print(torch.add(a,b))
res = torch.Tensor(4,3)
torch.add(a,b,out=res)
print(res)
a.add_(b)
print(a)

print(a[:,1])

c = torch.ones(7)
print(c)