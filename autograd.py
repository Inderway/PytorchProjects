import torch, torchvision

model = torchvision.models.resnet18(pretrained=True)
# one image with 3 channels, 64x64, means each channel is a 64x64 matrix
# dimension is 4
data = torch.rand(1, 3, 64, 64)

# a 1x1000 matrix, means 1000 labels
# dimension is 2
labels = torch.rand(1, 1000)

# make prediction
prediction = model(data)

# summarise the loss
loss = (prediction - labels).sum()

# back propagation
loss.backward()

# optimizer
optim = torch.optim.SGD(model.parameters((), lr=1e-2, momentum=0.9))  # learning rate=0.01

# initiate gradient descent
optim.step()


