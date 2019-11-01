# PyTorch

## PyTorchとは？

PyTorchは、オープンソースのPython ディープラーニングライブラリです。もともと、Facebookによって開発されていたものです。

公式サイトでは、PyTorchを以下のように説明しています。

> An open source machine learning framework that accelerates the path from research prototyping to production deployment.
>
>　訳： 研究用プロトタイプからプロダクト開発までのパスを加速させる、オープンソースの機械学習フレームワーク。

ディープラーニングを書くためのライブラリはいくつか存在します。GoogleのTensorFlowやその高レベルAPIであるKeras、Preffered NetworkdsのChainerが有名です。この中で近年人気を増やしているのがPyTorchです。[Google Trendsによる検索数の比較](https://trends.google.co.jp/trends/explore?geo=JP&q=PyTorch,TensorFlow,Chainer)を見てみましょう。検索数が一番多いのはTensorFlowで2番目がKersaで３番目がPyTorchです。徐々にTensorFlowや Kerasの検索数が減っているのに対し、PyTorchの検索数は徐々に増えているのがみて取れます。近年になってPyTorchの人気が高まっているのがわかります。

PyTorchの人気の秘密は、その書きやすさとプロダクトへの運用のしやすさのバランスにあると思います。たとえば、Kerasは非常に書きやすいライブラリですが、実運用にはやや工夫が必要になります。TensorFlowは、2.0になってからやや書きやすくなったものの、複雑なAPIを操作する必要があります。PyTorchはその中で現実的なバランスを持っています。ある程度の書きやすさを維持したまま、運用のしやすさを両立しています。
