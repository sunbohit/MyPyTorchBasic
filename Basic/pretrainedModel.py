import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

resnet = torchvision.models.resnet18(pretrained=True)

for param in resnet.parameters():
	param.requires_grad = False

resnet.fc = nn.Linear(resnet.fc.in_features, 100)

images = Variable(torch.randn(10, 3, 256, 256))
outputs = resnet(images)
print(outputs.size())

torch.save(resnet, 'model.pkl')
model = torch.load('model.pkl')

torch.save(resnet.state_dict(), 'params.pkl')
resnet.load_state_dict(torch.load('params.pkl'))