import torch
import torch.nn as nn
from torchinfo import summary
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from torchvision.transforms import ToTensor

data_train = torchvision.datasets.MNIST('letter_vision_model', download = True, train = True, transform = ToTensor())
data_test = torchvision.datasets.MNIST('letter_vision_model', download = True, train = False, transform= ToTensor())

# fig, ax = plt.subplots(1,7)
# for i in range(7):
#     ax[i].imshow(data_train[i][0].view(28,28))
#     ax[i].set_title(data_train[i][1])
#     ax[i].axis('off')

print('Training samples:',len(data_train))
print('Test samples:',len(data_test))

print('Tensor size:',data_train[0][0].size())
print('First 10 digits are:', [data_train[i][1] for i in range(10)])

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,100),
    nn.ReLU(),
    nn.Linear(100,10),
    nn.LogSoftmax(dim=0)
)

print("Digit to be predicted: ", data_train[0][1])
torch.exp(net(data_train[0][0]))

train_loader = torch.utils.data.DataLoader(data_train, batch_size = 64)
test_loader = torch.utils.data.DataLoader(data_test, batch_size = 64)

def train_epoch(net, dataloader, lr = 0.01, optimizer = None, loss_function = nn.NLLLoss()):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr = lr)
    net.train()
    total_loss = 0
    acc = 0
    count = 0
    for features, labels, in dataloader:
        optimizer.zero_grad()
        out = net(features)
        loss = loss_function(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
        _,predicted = torch.max(out,1)
        acc += (predicted==labels).sum()
        count += len(labels)
    return total_loss.item()/count, acc.item()/count

print(train_epoch(net, train_loader))

# weight_tensor = next(net.parameters())
# fig,ax = plt.subplots(1,10,figsize=(15,4))
# for i,x in enumerate(weight_tensor):
#     ax[i].imshow(x.view(28,28).detach())
# plt.show()

def validate(net, dataloader, loss_function = nn.NLLLoss()):
    net.eval()
    count, acc, loss = 0,0,0
    with torch.no_grad():
        for features, labels in dataloader:
            out = net(features)
            loss += loss_function(out,labels)
            pred = torch.max(out, 1)[1]
            acc += (pred == labels).sum()
            count += len(labels)
    return loss.item()/count, acc.item()/count

validate(net, test_loader)

def train(net, train_loader, test_loader, optimizer = None, lr = 0.01, epochs = 10, loss_function = nn.NLLLoss()):
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)
    res = { 'train_loss' : [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for ep in range(epochs):
        tl,ta = train_epoch(net,train_loader,optimizer=optimizer,lr=lr,loss_function=loss_function)
        vl,va = validate(net,test_loader,loss_function=loss_function)
        print(f"Epoch {ep:2}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}")
        res['train_loss'].append(tl)
        res['train_acc'].append(ta)
        res['val_loss'].append(vl)
        res['val_acc'].append(va)
    return res

hist = train(net, train_loader, test_loader, epochs=5)

plt.figure(figsize=(15,5))
plt.subplot(121)
plt.plot(hist['train_acc'], label='Training acc')
plt.plot(hist['val_acc'], label='Validation acc')
plt.legend()
plt.subplot(122)
plt.plot(hist['train_loss'], label='Training loss')
plt.plot(hist['val_loss'], label='Validation loss')
plt.legend()
plt.show()