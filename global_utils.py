import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)

def get_backbone(name, state_dict=None):

    if name == 'conv4':
        from backbones import conv4
        backbone = conv4()
    elif name == 'resnet12':
        from backbones import resnet12
        backbone = resnet12()
    elif name == 'resnet12_wide':
        from backbones import resnet12_wide
        backbone = resnet12_wide()
    elif name == 'resnet10':
        from backbones import resnet10
        backbone = resnet10()
    elif name == 'resnet18':
        from backbones import resnet18
        backbone = resnet18()
    elif name == 'resnet34':
        from backbones import resnet34
        backbone = resnet34()
    elif name == 'resnet50':
        from backbones import resnet50
        backbone = resnet50()
    elif name == 'resnet101':
        from backbones import resnet101
        backbone = resnet101()
    elif name == 'resnet152':
        from backbones import resnet152
        backbone = resnet152()
    elif name == 'wrn28_10':
        from backbones import wrn28_10
        backbone = wrn28_10()
    else:
        raise ValueError('Non-supported Backbone.')

    if state_dict is not None:
        backbone.load_state_dict(state_dict)

    return backbone
    

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Averager_with_interval():

    def __init__(self, confidence=0.95):
        self.list = []
        self.confidence = confidence
        self.n = 0

    def add(self, x):
        self.list.append(x)
        self.n += 1

    def item(self, return_str=False):
        mean, standard_error = np.mean(self.list), scipy.stats.sem(self.list)
        h = standard_error * scipy.stats.t._ppf((1 + self.confidence) / 2, self.n - 1)
        if return_str:
            return '{0:.2f}; {1:.2f}'.format(mean * 100, h * 100)
        else:
            return mean


def count_acc(logits, labels):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(labels).float())


def set_reproducibility(seed=0):
    set_seed(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_outputs_c_h(backbone, image_len):
    c_dict = {
        'conv4': 64,
        'resnet12': 512,
        'resnet12_wide': 640,
        'resnet18': 512,
        'wrn28_10': 640,
        'resnet10': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048
    }
    c = c_dict[backbone]

    h_devisor_dict = {
        'conv4': 16,
        'resnet12': 16,
        'resnet12_wide': 16,
        'resnet18': 8,
        'wrn28_10': 4,
        'resnet10': 8,
        'resnet34': 8,
        'resnet50': 8,
        'resnet101': 8,
        'resnet152': 8
    }

    h = image_len // h_devisor_dict[backbone]
    if image_len == 84 and h_devisor_dict[backbone] == 8:
        h = 11

    return c, h


def get_semantic_size(args):

    semantic_size_list = []

    for semantic_type in args.semantic_type:

        if semantic_type == 'class_name_embeddings':
            if args.train_data == 'miniimagenet':
                semantic_size_list.append(300)
            elif args.train_data == 'tieredimagenet':
                semantic_size_list.append(300)
        elif semantic_type == 'class_attributes':
            if args.train_data == 'miniimagenet':
                semantic_size_list.append(231)
            elif args.train_data == 'cub':
                semantic_size_list.append(312)
        elif semantic_type == 'image_attributes':
            if args.train_data == 'sun':
                semantic_size_list.append(102)
        elif semantic_type == 'class_description_glove':
            semantic_size_list.append(300)
        elif semantic_type == 'image_description_glove':
            semantic_size_list.append(300)
        elif semantic_type == 'class_description_bert':
            pass
        elif semantic_type == 'image_description_bert':
            pass

    if not len(semantic_size_list) == len(args.semantic_type):
        raise ValueError('Non-supported Semantic Type to the Dataset.')
    if len(semantic_size_list) == 1:
        return semantic_size_list[0]
    else:
        return semantic_size_list


def get_inputs_and_outputs(args, batch):

    semantic_type_limitation = [
        'class_name_embeddings',
        'class_attributes',
        'image_attributes',
        'class_description_glove',
        'class_description_bert',
        'image_description_glove',
        'image_description_bert',
    ]

    for semantic_type in args.semantic_type:
        if semantic_type not in semantic_type_limitation:
            raise ValueError('Non-supported Semantic Type.')

    return_list = ['images', 'targets'] + args.semantic_type

    if args.use_cuda:
        return [batch[return_type].cuda(non_blocking=True) for return_type in return_list]
    else:
        return [batch[return_type] for return_type in return_list]


def get_harmonic_mean(seen_acc, unseen_acc):
    if seen_acc == 0 and unseen_acc == 0:
        return 0
    else:
        return 2 * (seen_acc * unseen_acc) / (seen_acc + unseen_acc)
    
class MLP(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: Int, Input dimension
        out-dim: Int, Output dimension
        num_layer: Number of hidden layers
        relu: Bool, Use non linear function at output
        bias: Bool, Use bias
    '''
    def __init__(self, inp_dim, out_dim, num_layers = 1, relu = True, bias = True, dropout = False, norm = False, layers = []):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers.pop(0)
            mod.append(nn.Linear(incoming, outgoing, bias = bias))
            
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
                # mod.append(nn.BatchNorm1d(outgoing))
            mod.append(nn.ReLU(inplace = True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            if dropout:
                mod.append(nn.Dropout(p = 0.5))

        mod.append(nn.Linear(incoming, out_dim, bias = bias))

        if relu:
            mod.append(nn.ReLU(inplace = True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        self.mod = nn.Sequential(*mod)
    
    def forward(self, x):
        return self.mod(x)
    
    
class Meta_MLP(MetaModule):
    '''
    Baseclass to create a meta MLP
    Inputs
        inp_dim: Int, Input dimension
        out-dim: Int, Output dimension
        num_layer: Number of hidden layers
        relu: Bool, Use non linear function at output
        bias: Bool, Use bias
    '''
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(Meta_MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers.pop(0)
            mod.append(MetaLinear(incoming, outgoing, bias=bias))
            
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
                # mod.append(nn.BatchNorm1d(outgoing))
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            if dropout:
                mod.append(nn.Dropout(p=0.5))

        mod.append(MetaLinear(incoming, out_dim, bias=bias))

        if relu:
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        self.mod = MetaSequential(*mod)
    
    def forward(self, x, params=None):
        return self.mod(x, params=self.get_subdict(params, 'mod'))
    
