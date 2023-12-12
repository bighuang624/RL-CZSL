import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import json
import os
import io
import h5py

def reorganize(root):

    # Here we delete three anomalies in the CGQA dataset
    # ['2341198', '3', 'white', 'dry', 'erase_board.jpg']
    # ['2364112', '0', 'dry', 'dry', 'erase_board.jpg']
    # ['2409014', '22', 'white', 'dry', 'erase_board.jpg']
    os.remove(os.path.join(root, '2341198-3-white-dry-erase_board.jpg'))
    os.remove(os.path.join(root, '2364112-0-dry-dry-erase_board.jpg'))
    os.remove(os.path.join(root, '2409014-22-white-dry-erase_board.jpg'))

    for i in os.listdir(root):
        t = i.split('-')
        assert len(t) == 4
        s = t[2] + '_' + t[3].replace('.jpg', '')
        if s not in os.listdir(root):
            os.makedirs(os.path.join(root, s))
        os.rename(os.path.join(root, i), os.path.join(os.path.join(root, s, i)))


class CGQA(Dataset):

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, download=False):
        super(CGQA, self).__init__()

        self.folder = "cgqa"  # Q: This should go under the same folder as cub, i.e. assets
        self.filename = '{0}_data.hdf5'
        self.filename_labels = '{0}_labels.json'
        self.image_folder = 'images'

        if meta_train:
            self.meta_split = 'train'
        elif meta_val:
            self.meta_split = 'val'
        elif meta_test:
            self.meta_split = 'test'
        else:
            self.meta_split = meta_split
        self.transform = transform

        self.root = os.path.join(os.path.expanduser(root), self.folder)

        self.split_filename = os.path.join(self.root,
                                           self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
                                                  self.filename_labels.format(self.meta_split))

        if download:
            self.download()

        self.label_set, self.label_1_set, self.label_2_set = self.get_label_set()
        self._data_file = None
        self.data, self.labels, self.labels_1, self.labels_2 = self.get_data()

        # print('self.labels:', self.labels)

        if not self._check_integrity():
            raise RuntimeError('Test integrity check failed')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = {}

        image = Image.open(io.BytesIO(self.data[index])).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        item['images'] = image
        item['labels'] = self.labels[index]
        item['labels_1'] = self.labels_1[index]
        item['labels_2'] = self.labels_2[index]

        return item

    @property
    def num_classes(self):
        return len(self.label_set)

    def get_data(self):
        self._data_file = h5py.File(self.split_filename, 'r')
        data = list()
        labels = list()
        labels_1 = list()
        labels_2 = list()
        for label in self.label_set:
            temp = label.split('_')
            attr = temp[0]
            obj = temp[1]
            for img in self._data_file['datasets'][label]:
                data.append(img)
                labels.append(label)
                labels_1.append(attr)
                labels_2.append(obj)
        return data, labels, labels_1, labels_2

    def get_label_set(self):
        with open(self.split_filename_labels, 'r') as f:
            label_set = json.load(f)
        attrs = list()
        objs = list()
        for label in label_set:
            temp = label.split('_')
            attrs.append(temp[0])
            objs.append(temp[1])
        return label_set, attrs, objs

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
                and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self.data = None
            self.labels_1 = None
            self.labels_2 = None

    def download(self):
        import tarfile
        import zipfile
        import glob
        from tqdm import tqdm
        from torchvision.datasets.utils import download_url
        from sys import platform
        from api.datasets.utils import get_asset
        # f = "https://s3.mlcloud.uni-tuebingen.de/czsl/cgqa.zip"
        # # download_url(f, self.root)
        #
        # https://stackoverflow.com/questions/40419395/python-zipfile-extractall-ioerror-on-windows-when-extracting-files-from-long-pat
        # There is a path length problem, likely only in windows
        if 'win' in platform:  # Special treatment for windows processing systems
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

        #     # We extract the zip files here
        #     if 'zip' in f and 'images' not in os.listdir(self.root):
        #         ZipfileLongPaths(os.path.join(self.root, f.split('/')[-1])).extractall(self.root)
        #
        # else:  # Linux is fine with super-long file names
        #     if 'zip' in f and 'images' not in os.listdir(self.root):
        #         with zipfile.ZipFile(os.path.join(self.root, f.split('/')[-1]), 'r') as f:
        #             f.extractall(self.root)
        #     elif 'tar' in f and 'images' not in os.listdir(self.root):
        #         with tarfile.open(os.path.join(self.root, f.split('/')[-1]), 'r') as f:
        #             f.extractall(self.root)
        #
        image_folder = os.path.join(self.root, self.image_folder)
        # # reorganize(image_folder)
        #
        # # We have to process file names to replace ' ' with '_'
        # for i in os.listdir(image_folder):
        #     if os.path.isdir(os.path.join(image_folder, i)) and len(i.split(' ')) > 1:
        #         temp = i.split(' ')[0] + '_' + i.split(' ')[1]
        #         os.rename(os.path.join(image_folder, i), os.path.join(image_folder, temp))

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            if os.path.isfile(filename):
                print("The data has already been generated. If you did not make any changes, please proceed with interruption. Otherwise, please manually "
                      "delete the data under" + filename)
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
                        if 'win' in platform:
                            with open(winapi_path(image), 'rb') as f:
                                array = bytearray(f.read())
                                dataset[i] = np.asarray(array, dtype=np.uint8)
                        else:
                            with open(image, 'rb') as f:
                                array = bytearray(f.read())
                                dataset[i] = np.asarray(array, dtype=np.uint8)