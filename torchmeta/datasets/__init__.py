from torchmeta.datasets.triplemnist import TripleMNIST
from torchmeta.datasets.doublemnist import DoubleMNIST
from torchmeta.datasets.cub import CUB, CUBNormalDataset
from torchmeta.datasets.cifar100 import CIFARFS, FC100
from torchmeta.datasets.miniimagenet import MiniImagenet, MiniImagenetNormalDataset
from torchmeta.datasets.omniglot import Omniglot
from torchmeta.datasets.tieredimagenet import TieredImagenet, TieredImagenetNormalDataset
from torchmeta.datasets.tcga import TCGA
from torchmeta.datasets.pascal5i import Pascal5i
from torchmeta.datasets.mit_states import MITStates

from torchmeta.datasets.sun import SUN
from torchmeta.datasets.semantic import CUBMM, MiniImagenetMM, TieredImagenetMM, SUNMM

from torchmeta.datasets import helpers

__all__ = [
    'TCGA',
    'Omniglot',
    'MiniImagenet',
    'TieredImagenet',
    'CIFARFS',
    'FC100',
    'CUB',
    'DoubleMNIST',
    'TripleMNIST',
    'Pascal5i',
    'helpers',
    'CUBNormalDataset',
    'MiniImagenetNormalDataset',
    'TieredImagenetNormalDataset',
    'CUBMM',
    'MiniImagenetMM',
    'TieredImagenetMM',
    'SUNMM',
    'SUN',
    'MITStates'
]
