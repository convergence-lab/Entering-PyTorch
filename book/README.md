# Entering PyTorch

Convergence Lab. 木村優志

[http://www.convergence-lab.com](http://www.convergence-lab.com)

[kimura@convergence-lab.com](mailto:kimura@convergence-lab.com)

## はじめに

ディープラーニングは急速な発展をとげ、今まさに世界を塗り替えようとしています。ディープラーニングによって、それ以前には不可能であった様々なことが実現しているからです。

- 人間を超える精度の画像認識が実現され、近い将来に医療画像診断の自動化が実現するでしょう。
- 音声認識や自然言語処理の発展は、自然な対話インターフェースを実現するでしょう。
- 強化学習の進化は、自動運転技術につながっていきます。

ディープラーニングの衝撃というと、AlexNetがImageNetのコンペティションでSVMベースの手法に10%以上の差をつけて一位になった事例が注目されます。しかし、私にとってのそれは音声認識分野のことでした。大学の研究室で音声認識を研究していた、2012年頃のことです。Geoffrey Hintonらの新しいニューラルネットワークベースの音声認識システムが高い精度を実現したことを、以下の論文で知りました。

Geoffrey Hinton Li Deng Dong Yu George Dahl Abdel-rahman Mohamed Navdeep Jaitly Andrew Senior Vincent Vanhoucke Patrick Nguyen Brian Kingsbury Tara Sainath, Deep Neural Networks for Acoustic Modeling in Speech Recognition, IEEE Signal Processing Magazine | November 2012, Vol 29: pp. 82-97
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/38131.pdf


当時、同じようにニューラルネットワークを用いた音声認識を研究していた私にとって、６層のニューラルネットワークを利用するというのは考えられない方法でした。無限のユニットを持つ３層のニューラルネットワークは万能関数近似性をもつことが知られています。当時は、３層以上のニューラルネットワークを使うことは意味がないと思われていました。そのため、４層のネットワークを使うと、それだけで研究会で質問責めにあうような風潮でした。ReLU関数の有効性が発見されていない当時では、４層以上のニューラルネットワークの学習は現実的に困難でした。そんな中、Hinton達は、ディープラーニングによって当時の最先端スコアを相対で30%も改善しました。この技術は必ず身につけなければならないと、当時の私は思ったものです。

いまでは、ディープラーニングは大きな発展をとげ、誰もが簡単に利用できるようになっています。この本では、PyTorchを用いたディープラーニングのプログラミングを学びます。ディープラーニングの技術を是非、身につけてください。

## 本書について

本書では、どのようにディープラーニングのプログラムを書くのか、なぜそのように書くのかを説明することを目指しています。

この本は、基本的なPythonプログラムを習得していることを前提に書かれています。もし、Pythonに不慣れなのでしたら、[Python チュートリアル](https://docs.python.org/ja/3/tutorial/)を、学習することをお勧めします。
また、解説中に、Numpy や Pandas といったパッケージを用います。これらについての解説は本書の範囲外です。別の資料を参考にしてください。

機械学習やディープラーニングそのものの解説も含みますが、限定的な内容に止めるつもりです。それらは他の資料を参考にすると良い本がたくさんあります。

この本で対象にするPyTorchのバージョンは1.3です。それ以外のバージョンでは検証を行なっていないため、解説のコードが動かないかもしれません。

データサイエンスでは、Jupyter Notebookがよく利用されます。本書では、Jupyter Notebookの使用は最低限に止めることにします。Jupyterは便利ですが、AIプログラムのデプロイを考えた場合には、通常の .pyファイルの方が扱いやすためです
