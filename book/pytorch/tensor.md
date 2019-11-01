## Tensorの操作

まず、必要なモジュールをimportします。

```python
import torch
from torch import nn
import torch.nn.functional as F
```

PyTorchのプログラムは Tensorクラスでテンソルを操作することによって行います。基本的な操作をりかいするためには ipythonが便利です。今回は ipythonを使いましょう。

```python
$ ipython
In [1]: import torch

In [2]: torch.Tensor([1, 2, 3])
Out[2]: tensor([1., 2., 3.])
```

このように、Tensorを作成できます。次は、Tensorどうしを足してみましょう。

```python
In [3]: a = torch.Tensor([1, 2, 3])

In [4]: b = torch.Tensor([4, 5, 6])

In [5]: a + b
Out[5]: tensor([5., 7., 9.])
```

同様に、様々な演算が可能です。

```python
In [6]: a - b
Out[6]: tensor([-3., -3., -3.])

In [7]: a * b
Out[7]: tensor([ 4., 10., 18.])

In [8]: a / b
Out[8]: tensor([0.2500, 0.4000, 0.5000])

In [9]: a ** 2
Out[9]: tensor([1., 4., 9.])
```

演算は要素ごとに行われているのがわかると思います。そのため、次のようにテンソルの形が違うとうまく演算できません。

```python
In [10]: c = torch.Tensor([7, 8, 9, 10])

In [11]: a + c
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-11-e81e582b6fa9> in <module>()
----> 1 a + c

RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 0
```

この、テンソルの形のことを shapeと呼びます。shapeは .shape とすることで確認できます。

```python
In [12]: a.shape
Out[12]: torch.Size([3])

In [13]: b.shape
Out[13]: torch.Size([3])

In [14]: c.shape
Out[14]: torch.Size([4])
```

また、要素番号を指定することで、要素にアクセスできます。

```python
In [17]: a[0]
Out[17]: tensor(1.)

In [18]: b[0]
Out[18]: tensor(4.)
```

スライスを使う事もできます。

```python
In [22]: c[1:3]
Out[22]: tensor([8., 9.])
```

２次元以上のTensorも定義できます。ここでは、.empty()をつかって、初期化されていない４次元のTensorを作ってみましょう。引数には shapeを指定します。

```python
In [23]: torch.empty([1, 2, 3, 4])
Out[23]: 
tensor([[[[7.5205e+28, 7.1758e+22, 1.4603e-19, 1.8888e+31],
          [1.6216e-19, 1.1022e+24, 5.9437e-02, 7.0374e+22],
          [1.8037e+28, 1.7917e+25, 1.7751e+28, 5.6752e-02]],

         [[9.1041e-12, 6.2609e+22, 4.7428e+30, 4.5918e-40],
          [1.7753e+28, 1.7093e+25, 3.0346e+32, 2.8940e+12],
          [7.5338e+28, 7.1758e+22, 1.4603e-19, 1.8888e+31]]]])
```

初期化されたTensorをshapeを指定して作成したい場合には、zeros()やones()を使います。

```python
In [24]: torch.zeros([2, 2])
Out[24]: 
tensor([[0., 0.],
        [0., 0.]])

In [25]: torch.ones([3, 3, 3])
Out[25]: 
tensor([[[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]]])
```

GPUで計算したい場合は、TensorをCUDAデバイスに転送します。


(TBA)