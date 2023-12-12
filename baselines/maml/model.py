import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
sys.path.append("..") 
sys.path.append("../..")

from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)
from backbone import meta_conv4, meta_resnet10, meta_resnet18, meta_resnet34, meta_resnet50, meta_resnet101, meta_resnet152


def get_backbone(backbone):
    if backbone == 'conv4':
        return meta_conv4(), 64
    elif backbone == 'resnet10':
        return meta_resnet10(), 512
    elif backbone == 'resnet18':
        return meta_resnet18(), 512
    elif backbone == 'resnet34':
        return meta_resnet34(), 512
    elif backbone == 'resnet50':
        return meta_resnet50(), 2048
    elif backbone == 'resnet101':
        return meta_resnet101(), 2048
    elif backbone == 'resnet152':
        return meta_resnet152(), 2048


class MetaCNN(MetaModule):
    def __init__(self, backbone, n_s):
        super(MetaCNN, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))    # here we use avg_pool to reduce the size

        self.backbone, self.hidden_size = get_backbone(backbone)

        self.classifier_1 = MetaLinear(self.hidden_size, n_s)
        self.classifier_2 = MetaLinear(self.hidden_size, n_s)

    def forward(self, inputs, params=None):
        features = self.backbone(inputs, params=self.get_subdict(params, 'backbone'))
        features = self.avgpool(features)    # reduce the size
        features = features.view((features.size(0), -1))
        logits_1 = self.classifier_1(features, params=self.get_subdict(params, 'classifier_1'))
        logits_2 = self.classifier_2(features, params=self.get_subdict(params, 'classifier_2'))
        return logits_1, logits_2
