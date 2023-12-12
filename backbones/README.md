# Backbones

Several backbone networks are provided:

* `con4.py`: Conv-4
* `resnet12.py`: ResNet-12
* `resnet18.py`: ResNet-18
* `wrn.py`: WRN-28-10

## Introduction

### Conv-4

This architecture is composed of 4 convolutional blocks. Each block comprises a 64-filter 3x3 convolution, a batch normalization layer, a ReLU nonlinearity and a 2x2 max-pooling layer.

Used in "Matching Networks for One Shot Learning, NIPS 2016" and "Prototypical Networks for Few-shot Learning, NIPS 2017".

In "DPGN: Distribution Propagation Graph Network for Few-shot Learning, CVPR 2020", the last two blocks also contain a dropout layer. This is not implemented in our code.

没有 dropout，没有进行特殊的初始化。[3] 将 ReLU 换成了 LeakyReLU，并且在后两层的 LeakyReLU 后接了 dropout_rate=0.4 的 Dropout。

### ResNet-12

目前按照 [4] 的实现。激活函数为 leaky_relu。

[2] 提供了一份特殊的实现，其中的一个 drop_block 暂时没看懂。看懂完了可以测试一下哪一版效果更好。

Used in "TADAM: Task dependent adaptive metric for improved few-shot learning, NIPS 2018"

### ResNet-18

包含 resnet10, resnet18, resnet34, resnet50, resnet101, resnet152。

### WRN-28-10

28-layer Wide Residual Network

在对卷积层进行初始化时，[1] 用的`kaiming_normal_`，而 [2] 用的`xavier_uniform`。目前先采用 [1] 的方案，之后看实验效果。


## Reference

[1] [WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py)

[2] [FEAT](https://github.com/Sha-Lab/FEAT/blob/master/model/networks/WRN28.py)

[3] [DPGN](https://github.com/megvii-research/DPGN/blob/master/backbone.py)

[4] [few-shot-meta-baseline](https://github.com/cyvius96/few-shot-meta-baseline/blob/master/models/resnet12.py)