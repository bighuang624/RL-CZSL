import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import json
import os
import io
import h5py

def reorganize(root):
    """
    Reorganize the HICODET dataset to resemble the MIT-States dataset
    root/verb_obj/img1.jpg
    root/verb_obj/img2.jpg
    root/verb_obj/img3.jpg
    ...
    """

    path = "../hico_list_hoi.txt"
    hoi = dict()
    with open(path, 'r') as f:
        for i in f:
            l = i.strip().strip('\n').split(' ')
            assert len(l) == 3
            if int(l[0]) not in hoi.keys():  # and l[2] != "no_interaction":
                hoi[int(l[0])] = [l[2].replace('_', ''), l[1].replace('_', '')]
    print(hoi)

    path = "../hico_20160224_det/annotations/trainval_hico.json"
    with open(path, 'r') as f:
        a = json.load(f)

    for i in a:
        if type(i['hoi_annotation']) == list:  # This is just a greedy choice
            j = i['hoi_annotation'][0]
        else:
            j = i['hoi_annotation']
        if type(j['hoi_category_id']) == list:  # This is just a greedy choice
            temp = j['hoi_category_id'][0]
        else:
            temp = j['hoi_category_id']

        t = hoi[temp][0] + '_' + hoi[temp][1]
        if t not in os.listdir(root):
            os.makedirs(os.path.join(root, t))
        os.rename(os.path.join(root, i['file_name']), os.path.join(root, t, i['file_name']))

class HICO(Dataset):

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, download=False):
        super(HICO, self).__init__()

        self.folder = "hico"  # Q: This should go under the same folder as cub, i.e. assets
        self.filename = '{0}_data.hdf5'
        self.filename_labels = '{0}_labels.json'
        self.image_folder = 'hico_20160224_det/new_images'

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

        # TODO: change this, although current link is google drive
        # f = "http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip"
        # download_url(f, self.root)

        # This is not necessary for hico right now
        # if 'win' in platform:  # Special treatment for windows processing systems
        #     def winapi_path(dos_path, encoding=None):
        #         path = os.path.abspath(dos_path)
        #
        #         if path.startswith("\\\\"):
        #             path = "\\\\?\\UNC\\" + path[2:]
        #         else:
        #             path = "\\\\?\\" + path
        #
        #         return path
        #
        #     class ZipfileLongPaths(zipfile.ZipFile):
        #
        #         def _extract_member(self, member, targetpath, pwd):
        #             targetpath = winapi_path(targetpath)
        #             return zipfile.ZipFile._extract_member(self, member, targetpath, pwd)
        #
        #     # We extract the zip files here
        #     if 'zip' in f and 'ut-zap50k-images' not in os.listdir(self.root):
        #         ZipfileLongPaths(os.path.join(self.root, f.split('/')[-1])).extractall(self.root)
        #
        # else:  # Linux is fine with super-long file names
        #     if 'zip' in f and 'ut-zap50k-images' not in os.listdir(self.root):
        #         with zipfile.ZipFile(os.path.join(self.root, f.split('/')[-1]), 'r') as f:
        #             f.extractall(self.root)
        #     elif 'tar' in f and 'ut-zap50k-images' not in os.listdir(self.root):
        #         with tarfile.open(os.path.join(self.root, f.split('/')[-1]), 'r') as f:
        #             f.extractall(self.root)
        #
        image_folder = os.path.join(self.root,self.image_folder)
        # if not os.path.isdir(image_folder): # This means we have not processed hico labels yet
        #     reorganize(image_folder)
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
                        with open(image, 'rb') as f:
                            array = bytearray(f.read())
                            dataset[i] = np.asarray(array, dtype=np.uint8)