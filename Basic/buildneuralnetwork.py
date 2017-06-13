import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork,self).__init__()
		self.conv_1 = nn.Conv2d(1,8,3)
		self.conv_2 = nn.Conv2d(8,16,3)
		self.fullconnect_1 = nn.Linear(16*3*3,128)
		self.fullconnect_2 = nn.Linear(128,64)
		self.fullconnect_3 = nn.Linear(64,10)

	def forward(self,x):
		x = F.max_pool2d(F.relu(self.conv_1(x)), (2,2))
		x = F.max_pool2d(F.relu(self.conv_2(x)), (2,2))
		x = x.view(-1, self.num_features(x))
		x = F.relu(self.fullconnect_1(x))
		x = F.relu(self.fullconnect_2(x))
		x = self.fullconnect_3(x)

	def num_features(self,x):
		num_feature = 1
		for dim in x.size()[1:]:
			num_feature *= dim
		return num_feature

neu = NeuralNetwork()
print(neu)

paras = list(neu.parameters())
print(len(paras))
for i in range(len(paras)):
	print(paras[i].size())