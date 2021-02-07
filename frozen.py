from torch import nn, optim
import torchvision

model = torchvision.models.resnet18(pretrained=True)

# freeze all the parameters in the network
# means that they will not be adjusted
for param in model.parameters():
    param.requires_grad = False

# the classifier is the last linear layer model.fc in resnet
# we replace it with a new linear layer
model.fc = nn.Linear(512, 10)  # 512 inputs and 10 outputs

# we only need to compute the gradient of model.fc's output with respect to its input
# optimize only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

