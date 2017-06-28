import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

input_size = 1
output_size = 1
num_epochs = 100
learning_rate = 0.001

x_train = np.array([[1.03], [2.02], [3.03], [4.09], [4.93], [6.168], 
					[6.779], [8.182], [8.59], [10.167], [11.042], 
					[11.791], [13.313], [13.997], [15.1]], dtype=np.float32)

y_train = np.array([[0.07], [0.76], [2.09], [3.19], [3.694], [4.573], 
					[6.366], [6.596], [7.53], [9.221], [9.827], 
					[11.465], [11.65], [12.904], [14.3]], dtype=np.float32)

class LinearRegression(nn.Module):
	def __init__(self, input_size, output_size):
		super(LinearRegression, self).__init__()
		self.linear = nn.Linear(input_size, output_size)  

	def forward(self, x):
		out = self.linear(x)
		return out

model = LinearRegression(input_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

for epoch in range(num_epochs):
	inputs = Variable(torch.from_numpy(x_train))
	targets = Variable(torch.from_numpy(y_train))

	optimizer.zero_grad()  
	outputs = model(inputs)
	loss = criterion(outputs, targets)
	loss.backward()
	optimizer.step()

	if (epoch+1) % 5 == 0:
		print ('Epoch [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, loss.data[0]))
		print(model.linear.weight)
		print(model.linear.bias)
		print(model(Variable(torch.Tensor([[7.0]]), requires_grad=True)).data.numpy())

predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'lr_model.pkl')
