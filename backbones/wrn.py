import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, 
                            stride=stride, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1,
                            stride=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, inputs):
        outputs = self.dropout(self.conv1(self.relu1(self.bn1(inputs))))
        outputs = self.conv2(self.relu2(self.bn2(outputs)))
        outputs += self.shortcut(inputs)

        return outputs


class WideResNet(nn.Module):
    '''
    28-layer Wide Residual Network
    '''
    def __init__(self, depth, widen_factor, dropout_rate):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        n_channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]

        # 1st conv before any network layer
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st layer
        self.layer1 = self._wide_layer(BasicBlock, n_channels[1], n, dropout_rate, stride=1)
        # 2nd layer
        self.layer2 = self._wide_layer(BasicBlock, n_channels[2], n, dropout_rate, stride=2)
        # 3nd layer
        self.layer3 = self._wide_layer(BasicBlock, n_channels[3], n, dropout_rate, stride=2)
        # batchnorm before global average pooling
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._init_conv()

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def _init_conv(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = F.relu(self.bn1(outputs))
        # outputs = self.avgpool(outputs)
        
        return outputs


def wrn28_10(**kwargs):
    return WideResNet(depth=28, widen_factor=10, dropout_rate=0., **kwargs)
