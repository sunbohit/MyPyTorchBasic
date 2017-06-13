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

d = c.numpy()
c.add_(3)
print(c)
print(d)

import numpy as np
e = np.ones(4)
f = torch.from_numpy(e)
np.add(e,1,out=e)
print(e)
print(f)

if torch.cuda.is_available():
	print("CUDA Available")
	f = f.cuda()
else:
	print('None CUDA')