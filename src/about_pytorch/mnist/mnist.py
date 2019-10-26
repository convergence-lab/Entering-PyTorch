import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

class Net(nn.Modules):
    def __init__(self):
        self.base_net = nn.Sequential(
            [
                nn.Conv2d(1, 20, 5, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(20, 40, 5, 1),
                nn.ReLU() 
            ]
        )
        self.classfier = nn.Sequential(
            [
                nn.Linear(4*4*50, 100),
                nn.ReLU()
                nn.Linear(100, 10)
                nn.LogSoftmax()
            ]
        )
    
    def forward(self, x):
        x = self.base_net(x)
        x = x.view(-1, 4*4*50)
        x = self.classfier(x)
        return x


def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    for batch in train_loader:
        data, target = batch
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch}: Train loss {train_loss / len(train_loader)}")

def test(model, test_loader, criterion)
    model.eval()
    test_loss = 0
    for batch in test_loader:
        data, target = batch
        pred = model(data)
        loss = criterion(pred, target)
        test_loss += loss.item()
    print(f"Epoch {epoch}: Test loss {train_loss / len(train_loader)}, Accuracy")
