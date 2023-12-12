import numpy as np
from PIL import Image
import os
import io
import json
import h5py
from sys import platform

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchmeta.datasets.utils import get_asset


class MITStates(CombinationMetaDataset):
    """

    Parameters
    ----------
    root : string
        Root directory where the dataset folder `mit_states` exists.

    num_classes_per_task : int
        Number of classes per tasks. This corresponds to "N" in "N-way"
        classification.

    meta_train : bool (default: `False`)
        Use the meta-train split of the dataset. If set to `True`, then the
        arguments `meta_val` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.

    meta_val : bool (default: `False`)
        Use the meta-validation split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.

    meta_test : bool (default: `False`)
        Use the meta-test split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_val` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.

    meta_split : string in {'train', 'val', 'test'}, optional
        Name of the split to use. This overrides the arguments `meta_train`,
        `meta_val` and `meta_test` if all three are set to `False`.

    transform : callable, optional
        A function/transform that takes a `PIL` image, and returns a transformed
        version. See also `torchvision.transforms`.

    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed
        version. See also `torchvision.transforms`.

    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a
        transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These classes
        are transformations of existing classes. E.g.
        `torchmeta.transforms.HorizontalFlip()`.

    download : bool (default: `False`)
        If `True`, downloads the pickle files and processes the dataset in the root
        directory (under the `cub` folder). If the dataset is already
        available, this does not download/process the dataset again.

    Notes here
    -----

    """

    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = MITStatesClassDataset(root, meta_train=meta_train, meta_val=meta_val,
                                        meta_test=meta_test, meta_split=meta_split, transform=transform,
                                        class_augmentations=class_augmentations, download=download)
        super(MITStates, self).__init__(dataset, num_classes_per_task,
                                        target_transform=target_transform, dataset_transform=dataset_transform)


class MITStatesClassDataset(ClassDataset):


    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(MITStatesClassDataset, self).__init__(meta_train=meta_train,
                                                    meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
                                                    class_augmentations=class_augmentations)

        self.folder = "mit_states" # Q: This should go under the same folder as cub, i.e. assets
        self.filename = '{0}_data.hdf5'
        self.filename_labels = '{0}_labels.json'
        self.image_folder = 'release_dataset/images'

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root,
                                           self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
                                                  self.filename_labels.format(self.meta_split))

        self._data_file = None
        self._data = None
        self.labels = None

        # Q: Currently this download function is just for data processing
        # self.download()
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Test integrity check failed')

        if self.labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self.labels = json.load(f)
            attr = list()
            objs = list()
            for i in self.labels:
                temp = i.split("_")
                attr.append(temp[0])
                objs.append(temp[1])
        self.labels = (attr, objs)

        # Q: Hypothesis: the __len__ function cannot return a non-integer number
        # Q: In this case, it makes more sense to not ask it to return a tuple
        assert len(self.labels[0]) == len(self.labels[1])
        self._num_classes = len(self.labels[0])

    def __getitem__(self, index):
        attr_label = self.labels[0][index % self.num_classes]
        objs_label = self.labels[1][index % self.num_classes]
        label = (attr_label, objs_label)
        data = self.data[attr_label + '_' + objs_label]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return MITStatesDataset(index, data, label, transform=transform,
                                target_transform=target_transform)

    # Q: I deleted label property because cannot unpack non-iterable NoneType object

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
                and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self):
        import tarfile
        import zipfile
        import glob
        from tqdm import tqdm
        from torchvision.datasets.utils import download_url
        f = "http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip"
        download_url(f, self.root)

        # https://stackoverflow.com/questions/40419395/python-zipfile-extractall-ioerror-on-windows-when-extracting-files-from-long-pat
        # There is a path length problem, likely only in windows
        if 'win' in platform: #Special treatment for windows processing systems
            def winapi_path(dos_path, encoding=None):
                path = os.path.abspath(dos_path)

                if path.startswith("\\\\"):
                    path = "\\\\?\\UNC\\" + path[2:]
                else:
                    path = "\\\\?\\" + path

                return path

            class ZipfileLongPaths(zipfile.ZipFile):

                def _extract_member(self, member, targetpath, pwd):
                    targetpath = winapi_path(targetpath)
                    return zipfile.ZipFile._extract_member(self, member, targetpath, pwd)

            # We extract the zip files here
            if 'zip' in f and 'release_dataset' not in os.listdir(self.root):
                ZipfileLongPaths(os.path.join(self.root, f.split('/')[-1])).extractall(self.root)

        else: #Linux is fine with super-long file names
            if 'zip' in f and 'release_dataset' not in os.listdir(self.root):
                with zipfile.ZipFile(os.path.join(self.root, f.split('/')[-1]), 'r') as f:
                    f.extractall(self.root)
            elif 'tar' in f and 'release_dataset' not in os.listdir(self.root):
                with tarfile.open(os.path.join(self.root, f.split('/')[-1]), 'r') as f:
                    f.extractall(self.root)

        image_folder = os.path.join(self.root, self.image_folder)

        # We have to process file names to replace ' ' with '_'
        for i in os.listdir(image_folder):
            if os.path.isdir(os.path.join(image_folder, i)) and len(i.split(' ')) > 1:
                temp = i.split(' ')[0] + '_' + i.split(' ')[1]
                os.rename(os.path.join(image_folder, i), os.path.join(image_folder, temp))

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            if os.path.isfile(filename):
                continue


            labels = get_asset(self.folder, '{0}.json'.format(split))
            labels_filename = os.path.join(self.root, self.filename_labels.format(split))
            with open(labels_filename, 'w') as f:
                json.dump(labels, f)

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                dtype = h5py.special_dtype(vlen=np.uint8)
                for i, label in enumerate(tqdm(labels, desc=filename)):
                    images = glob.glob(os.path.join(image_folder, label, '*.jpg'))
                    images.sort()
                    dataset = group.create_dataset(label, (len(images),), dtype=dtype)
                    for i, image in enumerate(images):
                        with open(image, 'rb') as f:
                            array = bytearray(f.read())
                            dataset[i] = np.asarray(array, dtype=np.uint8)

        # tar_folder, _ = os.path.splitext(tgz_filename)
        # if os.path.isdir(tar_folder):
        #     shutil.rmtree(tar_folder)

        # attributes_filename = os.path.join(self.root, 'attributes.txt')
        # if os.path.isfile(attributes_filename):
        #     os.remove(attributes_filename)


class MITStatesDataset(Dataset):
    def __init__(self, index, data, label,
                 transform=None, target_transform=None):
        super(MITStatesDataset, self).__init__(index, transform=transform,
                                               target_transform=target_transform)
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(io.BytesIO(self.data[index])).convert('RGB')
        target = self.label

        if self.transform is not None:
            image = self.transform(image)

        # Change this back. I commented this out because there was an index problem
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)