from tqdm import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.base_net = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, 1),
            nn.ReLU() 
        )
        self.classfier = nn.Sequential(
            nn.Linear(40*8*8, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.LogSoftmax()
        )
    
    def forward(self, x):
        x = self.base_net(x)
        x = x.view(-1, 40*8*8)
        x = self.classfier(x)
        return x


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader):
        data, target = batch
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch}: Train loss {train_loss / len(train_loader)}")

def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for batch in tqdm(test_loader):
        data, target = batch
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = criterion(pred, target)
        test_loss += loss.item()
        correct += pred.argmax(dim=1).eq(target).sum().item()
    print(f"Epoch {epoch}: Test loss {test_loss / len(test_loader)}, Accuracy {100. * correct / len(test_loader.dataset)} %")

def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch = 5
    batch_size = 100
    save_model = False

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    for ep in range(epoch):
        train(model, device, train_loader, optimizer, criterion, ep)
        test(model, device, test_loader, criterion, ep)

    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == "__main__":
    main()
