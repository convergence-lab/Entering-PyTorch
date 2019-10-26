# PyTorch

## PyTorchとは？

PyTorchは、オープンソースのPython ディープラーニングライブラリです。もともと、Facebookによって開発されていたものです。

公式サイトでは、PyTorchを以下のように説明しています。

> An open source machine learning framework that accelerates the path from research prototyping to production deployment.
>
>　訳： 研究用プロトタイプからプロダクト開発までのパスを加速させる、オープンソースの機械学習フレームワーク。

ディープラーニングを書くためのライブラリはいくつか存在します。GoogleのTensorFlowやその高レベルAPIであるKeras、Preffered NetworkdsのChainerが有名です。この中で近年人気を増やしているのがPyTorchです。[Google Trendsによる検索数の比較](https://trends.google.co.jp/trends/explore?geo=JP&q=PyTorch,TensorFlow,Chainer)を見てみましょう。検索数が一番多いのはTensorFlowで2番目がKersaで３番目がPyTorchです。徐々にTensorFlowや Kerasの検索数が減っているのに対し、PyTorchの検索数は徐々に増えているのがみて取れます。近年になってPyTorchの人気が高まっているのがわかります。

PyTorchの人気の秘密は、その書きやすさとプロダクトへの運用のしやすさのバランスにあると思います。たとえば、Kerasは非常に書きやすいライブラリですが、実運用にはやや工夫が必要になります。TensorFlowは、2.0になってからやや書きやすくなったものの、複雑なAPIを操作する必要があります。PyTorchはその中で現実的なバランスを持っています。ある程度の書きやすさを維持したまま、運用のしやすさを両立しています。

## PyTorchのプログラムを書いてみよう

実際のPyTorchのプログラムがどのようなものかをみていきましょう。

```python
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
```
