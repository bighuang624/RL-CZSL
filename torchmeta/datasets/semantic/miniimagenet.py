import numpy as np
from PIL import Image
import os
import io
import re
import json
import glob
import h5py
import torch
import pickle
import scipy.io

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import download_url, download_file_from_google_drive
from torchmeta.datasets.utils import get_asset



class MiniImagenetMM(CombinationMetaDataset):
    """
    The Mini-Imagenet dataset, introduced in [1]. This dataset contains images 
    of 100 different classes from the ILSVRC-12 dataset (Imagenet challenge). 
    The meta train/validation/test splits are taken from [2] for reproducibility.

    Parameters
    ----------
    root : string
        Root directory where the dataset folder `miniimagenet` exists.

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
        directory (under the `miniimagenet` folder). If the dataset is already 
        available, this does not download/process the dataset again.

    Notes
    -----
    The dataset is downloaded from [this repository]
    (https://github.com/renmengye/few-shot-ssl-public/). The meta train/
    validation/test splits are over 64/16/20 classes.

    References
    ----------
    .. [1] Vinyals, O., Blundell, C., Lillicrap, T. and Wierstra, D. (2016). 
           Matching Networks for One Shot Learning. In Advances in Neural 
           Information Processing Systems (pp. 3630-3638) (https://arxiv.org/abs/1606.04080)

    .. [2] Ravi, S. and Larochelle, H. (2016). Optimization as a Model for 
           Few-Shot Learning. (https://openreview.net/forum?id=rJY0-Kcll)
    """
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False, semantic_type=None):
        dataset = MiniImagenetClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, class_augmentations=class_augmentations,
            download=download, semantic_type=semantic_type)
        super(MiniImagenetMM, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class MiniImagenetClassDataset(ClassDataset):
    folder = 'miniimagenet'
    # Google Drive ID from https://github.com/renmengye/few-shot-ssl-public
    gdrive_id = '16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY'
    gz_filename = 'mini-imagenet.tar.gz'
    gz_md5 = 'b38f1eb4251fb9459ecc8e7febf9b2eb'
    pkl_filename = 'mini-imagenet-cache-{0}.pkl'

    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    # for label embedding (Glove)
    assets_folder = 'assets'
    class_name_filename = 'class_name.json'
    glove_filename = 'glove.840B.300d.txt'

    # for class attributes
    class_attribute_filename_labels = 'miniimagenet_attributes.json'

    # for class description
    class_description_filename = 'miniimagenet_class_description.json'
    class_description_word2id_filename = 'miniimagenet_word2id.npy'
    class_description_glove_filename = 'miniimagenet_glove_300d.npy'

    glove_embedding_dim = 300
    attributes_dim = 231

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False, semantic_type=None):
        super(MiniImagenetClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)
        
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform
        self.semantic_type = semantic_type
        self.semantic_type_limitation = [
            'class_name_embeddings',
            'class_attributes',
            'class_description_glove',
            'class_description_bert',
        ]
        if self.semantic_type is not None:
            for single_semantic_type in self.semantic_type:
                if single_semantic_type not in self.semantic_type_limitation:
                    raise ValueError('Non-supported Semantic Type for MiniImagenet.')

        self.split_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        self.class_name_filepath = os.path.join(os.path.expanduser(root), 
            self.assets_folder, self.folder, self.class_name_filename)
        self.glove_filepath = os.path.join(os.path.expanduser(root), 
            self.assets_folder, self.glove_filename)
        self.class_attribute_filepath = os.path.join(os.path.expanduser(root), 
            self.assets_folder, self.folder, self.class_attribute_filename_labels)
        self.class_description_filepath = os.path.join(os.path.expanduser(root), 
            self.assets_folder, self.folder, self.class_description_filename)
        self.class_description_word2id_filepath = os.path.join(os.path.expanduser(root), 
            self.assets_folder, self.folder, self.class_description_word2id_filename)
        self.class_description_glove_filepath = os.path.join(os.path.expanduser(root), 
            self.assets_folder, self.folder, self.class_description_glove_filename)

        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('MiniImagenet integrity check failed')
        self._num_classes = len(self.labels)

        if 'class_name_embeddings' in self.semantic_type:
            self.class_name_embedding_dict = self.get_class_name_embedding_dict()
        if 'class_attributes' in self.semantic_type:
            self.class_attributes_dict = self.get_class_attributes_dict()
        self.class_description_dict = None    # ? any better solution
        if 'class_description_glove' in self.semantic_type:
            self.class_description_glove_dict = self.get_class_description_glove_dict()
        # if 'class_description_bert' in self.semantic_type:
        #     self.class_description_bert_dict = self.get_class_description_bert_dict()

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        data = self.data[class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        semantics = {}
        if 'class_name_embeddings' in self.semantic_type:
            semantics['class_name_embeddings'] = self.class_name_embedding_dict[class_name]
        if 'class_attributes' in self.semantic_type:
            semantics['class_attributes'] = self.class_attributes_dict[class_name]
        if 'class_description_glove' in self.semantic_type:
            semantics['class_description_glove'] = self.class_description_glove_dict[class_name]
        # if 'class_description_bert' in self.semantic_type:
        #     semantics['class_description_bert'] = self.class_description_bert_dict[class_name]

        return MiniImagenetDataset(index, data, class_name, 
            self.semantic_type, semantics,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels)
            and os.path.isfile(self.class_name_filepath)
            and os.path.isfile(self.glove_filepath))

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def get_glove_word_embedding_dict(self):

        print('Start to read pre-trained Glove embedding')
        
        # load word embedding dict
        word_embedding_dict = {}
        word_embedding_file = io.open(self.glove_filepath, 'r', encoding='utf-8', newline='\n', errors='ignore')
        for line in word_embedding_file:
            tokens = line.rstrip().split(' ')
            tmpv = [float(tokens[i]) for i in range(1, len(tokens))]
            word_embedding_dict[tokens[0]] = tmpv

        return word_embedding_dict

    def get_class_name_embedding_dict(self):

        word_embedding_dict = self.get_glove_word_embedding_dict()

        print('Start to get class name embedding dict')
        # here we use a id_to_real_class_name file different from AM3: https://github.com/ElementAI/am3/blob/master/datasets/create_dataset_miniImagenet.py
        # we choose to delete all ',' before splitting
        class_name_embedding_dict = {}
        with open(self.class_name_filepath, 'r') as f:
            class_name_dict = json.load(f)
        for class_key in class_name_dict.keys():
            if class_key not in class_name_embedding_dict.keys():
                class_names = class_name_dict[class_key].lower().split(', ')
                tmpv = np.zeros(self.glove_embedding_dim)
                tmpl = []
                count = 0
                class_name_words = class_names[0].split(' ')    # only use the first name
                for word in class_name_words:
                    if word in word_embedding_dict.keys():
                        tmpv += word_embedding_dict[word]
                        tmpl.append(word)
                        count += 1
                if count != 0:
                    class_name_embedding_dict[class_key] = tmpv / count
                else:
                    class_name_embedding_dict[class_key] = tmpv    # zero vectors

        del word_embedding_dict
        return class_name_embedding_dict


    # def text_preprocess(self, sentence):
    #     stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

    #     delete_words = [',', '.', '?', '!', ';', '(', ')', '/', '\xef\xbf\xbd']

    #     blank_words = ['-', '\"', '\'']

    #     for delete_word in delete_words:
    #         sentence = sentence.replace(delete_word, '')
        
    #     for blank_word in blank_words:
    #         sentence = sentence.replace(blank_word, ' ')

    #     final_words = []
    #     sentence_words = sentence.split(' ')
    #     for word in sentence_words:
    #         word = word.lower()
    #         if word not in stop_words:
    #             final_words.append(word)

    #     del sentence_words
    #     return final_words

    def clean_str(self, string):
        """
        Tokenization/string cleaning.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()
    

    def get_class_description_dict(self):
        with open(self.class_description_filepath, 'r') as f:
            class_description_dict = json.load(f)
        # clean and split string into words here
        for class_name in class_description_dict.keys():
            # print('descrption:', class_description_dict[class_name])
            # print('clean descrption:', clean_str(self.class_description_dict[class_name]))
            # print('split clean descrption:', self.clean_str(self.class_description_dict[class_name]).split())
            class_description_dict[class_name] = self.clean_str(class_description_dict[class_name]).split()
        return class_description_dict


    def get_class_description_glove_dict(self):
        if self.class_description_dict is None:
            self.class_description_dict = self.get_class_description_dict()

        word2id = np.load(self.class_description_word2id_filepath, allow_pickle=True).item()
        glove_list = np.load(self.class_description_glove_filepath, allow_pickle=True)

        # print('type word2id', type(word2id))
        # print('word2id', word2id)

        # calculate the max_len
        max_len = max(map(lambda x: len(self.class_description_dict[x]), self.class_description_dict.keys()))

        # get embeddings with Glove
        # ! ? 是直接用预训练好的，还是导出 index 用 nn.Embedding 训练？后者跨域时好像不好使？先用前者试试，后者可以再写一个接口
        class_description_glove_dict = {}
        for class_name in self.class_description_dict.keys():
            word_id_list = list(map(lambda w: word2id.get(w, 0), self.class_description_dict[class_name]))
            word_id_list = word_id_list + [len(word_id_list)-1] * (max_len - len(word_id_list))
            word_glove_list = np.array(list(map(lambda x: glove_list[x], word_id_list)))
            class_description_glove_dict[class_name] = word_glove_list

        return class_description_glove_dict

    
    # def get_class_description_bert_dict(self):
    #     if not self.class_description_dict:
    #         self.class_description_dict = self.get_class_description_dict()
    #     pass


    def get_class_attributes_dict(self):
        with open(self.class_attribute_filepath, 'r') as f:
            class_attributes_dict = json.load(f)
            for class_name in class_attributes_dict.keys():
                class_attributes_dict[class_name] = np.array(class_attributes_dict[class_name])
        return class_attributes_dict


    def download(self):
        import tarfile

        if self._check_integrity():
            return

        download_file_from_google_drive(self.gdrive_id, self.root,
            self.gz_filename, md5=self.gz_md5)

        filename = os.path.join(self.root, self.gz_filename)
        with tarfile.open(filename, 'r') as f:
            f.extractall(self.root)

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            if os.path.isfile(filename):
                continue

            pkl_filename = os.path.join(self.root, self.pkl_filename.format(split))
            if not os.path.isfile(pkl_filename):
                raise IOError()
            with open(pkl_filename, 'rb') as f:
                data = pickle.load(f)
                images, classes = data['image_data'], data['class_dict']

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                for name, indices in classes.items():
                    group.create_dataset(name, data=images[indices])

            labels_filename = os.path.join(self.root, self.filename_labels.format(split))
            with open(labels_filename, 'w') as f:
                labels = sorted(list(classes.keys()))
                json.dump(labels, f)

            if os.path.isfile(pkl_filename):
                os.remove(pkl_filename)


class MiniImagenetDataset(Dataset):
    def __init__(self, index, data, class_name, semantic_type, semantics,
                 transform=None, target_transform=None):
        super(MiniImagenetDataset, self).__init__(index, transform=transform,
                                                  target_transform=target_transform)
        self.data = data
        self.class_name = class_name
        self.semantic_type = semantic_type
        self.content = semantics

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        self.content['images'] = image
        self.content['targets'] = target

        return self.content
