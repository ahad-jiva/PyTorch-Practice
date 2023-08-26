import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(root="pytorch vision model", train = True, download = True, transform = ToTensor())
test_data = datasets.FashionMNIST(root="pytorch vision model", train = False, download = True, transform=ToTensor())

train_dataloader = DataLoader(training_data, batch_size = 64)
test_dataloader = DataLoader(test_data, batch_size = 64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
    )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

learning_rate = 1e-2
batch_size = 64
epochs = 10

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

def training_loop(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        predicted = model(X)
        loss = loss_function(predicted, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}    [{current:>5d}/{size:>5d}]')


def test_loop(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            predicted = model(X)
            test_loss += loss_function(predicted, y).item()
            correct += (predicted.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n--------------------------------")
    training_loop(train_dataloader, model, loss_function, optimizer)
    test_loop(test_dataloader, model, loss_function)
print("Done!")

torch.save(model.state_dict(), "pytorch vision model/vision_model.pth")
print("Saved model state")