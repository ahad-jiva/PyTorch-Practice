import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchinfo import summary
import numpy as np
from letter_recognition import train_loader, test_loader

class OneConv(nn.Module):
    def __init__(self):
        super(OneConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels= 9, kernel_size=(5,5))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(5184, 10)
    def forward(self, x):
        x = nn.functional.relu(self.conv(x))
        x = self.flatten(x)
        x = nn.functional.log_softmax(self.fc(x),dim=1)
        return x
    
net = OneConv()

summary(net, input_size=(1,1,28,28))

# fig,ax = plt.subplots(1,9)
# with torch.no_grad():
#     p = next(net.conv.parameters())
#     for i,x, in enumerate(p):
#         ax[i].imshow(x.detach().cpu()[0,...])
#         ax[i].axis('off')

class MultiLayerCNN(nn.Module):
    def __init__(self):
        super(MultiLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1,10,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(10,20,5)
        self.fc = nn.Linear(320,10)
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 320)
        x = nn.functional.log_softmax(self.fc(x), dim =1)
        return x
    
    
net = MultiLayerCNN()
summary(net, input_size=(1,1,28,28))