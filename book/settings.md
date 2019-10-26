# 環境設定

本書のコードを動かすための環境設定方法について解説します。ここでは、Anacondaを利用した環境設定方法を説明します。

## Anaconda

Anacondaは、Pythonのディストリビューターの一つです。科学計算やデータサイエンス、AIプログラミングに必要なパッケージを利用することができます。

[Anaconda Distribution](https://www.anaconda.com/distribution/)のページから、ご自身のOSにあった Anacondaをインストールしてください。 Python 3.x と Python 2.7が選べますが、ここでは3系のAnacodaを用います。

## Pytorchのインストール

Pytorchそのものは、[PytorchのGET STARGED](https://pytorch.org/get-started/locally/)ページのとおりコマンドを書くことでインストールできます。環境に合ったコマンドを選択してください。


### CPU版（Mac, Linux)

例えば、MaxやLinuxでCPU版の PyTorchをインストールするには、次のコマンドを入力します。


```
conda install pytorch torchvision -c pytorch
```

### CPU版 (Widnows)

Windwosでは次のコマンドを入力します。

```
conda install pytorch torchvision cpuonly -c pytorch
```

### GPU版　（Linux、Windowsの場合）

GPU版のPytorchをインストールするためには次のコマンドを入力します。MacOSはCUDAをサポートしていないため、GPU版のPyTorchをインストールすることはできません。

```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```


(TBA)