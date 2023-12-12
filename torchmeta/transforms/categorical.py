import torch
from torchmeta.transforms.utils import apply_wrapper
from collections import defaultdict

from torchmeta.transforms.target_transforms import TargetTransform


class Categorical(TargetTransform):
    """Target transform to return labels in `[0, num_classes)`.

    Parameters
    ----------
    num_classes : int, optional
        Number of classes. If `None`, then the number of classes is inferred
        from the number of individual labels encountered.

    Examples
    --------
    >>> dataset = Omniglot('data', num_classes_per_task=5, meta_train=True)
    >>> task = dataset.sample_task()
    >>> task[0]
    (<PIL.Image.Image image mode=L size=105x105 at 0x11EC797F0>,
    ('images_evaluation/Glagolitic/character12', None))

    >>> dataset = Omniglot('data', num_classes_per_task=5, meta_train=True,
    ... target_transform=Categorical(5))
    >>> task = dataset.sample_task()
    >>> task[0]
    (<PIL.Image.Image image mode=L size=105x105 at 0x11ED3F668>, 2)
    """
    def __init__(self, num_classes=None):
        super(Categorical, self).__init__()
        self.num_classes = num_classes #Q: common between our labels
        self._attr_classes = None
        self._objs_classes = None
        self._attr_labels = None
        self._objs_labels = None

    def reset(self):
        self._attr_classes = None
        self._objs_classes = None
        self._attr_labels = None
        self._objs_labels = None

    @property
    def attr_classes(self):
        if self._attr_classes is None:
            self._attr_classes = defaultdict(None)
            if self.num_classes is None:
                default_factory = lambda: len(self._attr_classes)
            else:
                default_factory = lambda: self.attr_labels[len(self._attr_classes)]
            self._attr_classes.default_factory = default_factory
        if (self.num_classes is not None) and (len(self._attr_classes) > self.num_classes):
            raise ValueError('The number of individual labels ({0}) is greater '
                'than the number of classes defined by `num_classes` '
                '({1}).'.format(len(self._attr_classes), self.num_classes))
        return self._attr_classes

    @property
    def objs_classes(self):
        if self._objs_classes is None:
            self._objs_classes = defaultdict(None)
            if self.num_classes is None:
                default_factory = lambda: len(self._objs_classes)
            else:
                default_factory = lambda: self.objs_labels[len(self._objs_classes)]
            self._objs_classes.default_factory = default_factory
        if (self.num_classes is not None) and (len(self._objs_classes) > self.num_classes):
            raise ValueError('The number of individual labels ({0}) is greater '
                'than the number of classes defined by `num_classes` '
                '({1}).'.format(len(self._objs_classes), self.num_classes))
        return self._objs_classes

    @property
    def attr_labels(self):
        if (self._attr_labels is None) and (self.num_classes is not None):
            # TODO: Replace torch.randperm with seed-friendly counterpart
            self._attr_labels = torch.randperm(self.num_classes).tolist()
        return self._attr_labels

    @property
    def objs_labels(self):
        if (self._objs_labels is None) and (self.num_classes is not None):
            # TODO: Replace torch.randperm with seed-friendly counterpart
            self._objs_labels = torch.randperm(self.num_classes).tolist()
        return self._objs_labels

    def __call__(self, target):
        # print(self.attr_classes)
        # print(self.objs_classes)
        # print("target = ", target)
        return (self.attr_classes[target[0][0]], self.objs_classes[target[0][1]])

    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, self.num_classes or '')


class FixedCategory(object):
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, index):
        return (index, self.transform)

    def __repr__(self):
        return ('{0}({1})'.format(self.__class__.__name__, self.transform))
