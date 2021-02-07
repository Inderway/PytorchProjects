import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        # 1 sample, 6 channels
        self.conv1=nn.Conv2d(1,6,3)
        self.conv2=nn.Conv2d(6,16,3)
        # an affine operation: y=Wx+b
        self.fc1=nn.Linear(16*6*6,120) # 6*6 from image dimension ?
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        # max pooling over a 2x2 window1
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        # if the size is a square you can only specify a single number instead of (2,2)
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(-1,self.num_flat_features(x)) # dispute x in x.size/num_flat_features batches, each has num_flat_features data
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension (x[0]) ?
        num_features=1
        for s in size:
            num_features*=s
        return num_features

net=Net()
print(net)
params=list(net.parameters())
print(len(params))
print(params[1].size()) # params[0] is conv1's weight

input=torch.randn(1,1,32,32) # obey normal disputation
out=net(input)
print(out)

# zero the gradient buffers
net.zero_grad()
# back prop with random gradients
out.backward(torch.randn(1,10))

output=net(input)
target=torch.randn(10) # a dummy target
target=target.view(1,-1) # make it the same shape as output (1 batch, 10 data, a 1x10 tensor)
# previous target was [...], it becomes [[...]] after view
criterion=nn.MSELoss() # mean-squared error

loss=criterion(output,target)
print(loss)

# to backpropagate the error, existing gradients should be cleared
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# create your optimizer
optimizer=optim.SGD(net.parameters(),lr=0.01)

# in your training loop
optimizer.zero_grad()
output=net(input)
loss=criterion(output,target)
loss.backward()
optimizer.step() # do the update
