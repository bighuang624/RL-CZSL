import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

import sys 
sys.path.append("..") 
sys.path.append("../..") 


def get_dataloader(args, dataset_name, phase):

    if dataset_name == 'mit':
        from api.datasets import MITStates as CGDataset
        image_size = 84
        padding_len = 8
    elif dataset_name == 'ut':
        from api.datasets import UTZap as CGDataset
        image_size = 84
        padding_len = 8
    elif dataset_name == 'hico':
        from api.datasets import HICO as CGDataset
        image_size = 84
        padding_len = 8
    elif dataset_name == 'cgqa':
        from api.datasets import CGQA as CGDataset
        image_size = 84
        padding_len = 8
    elif dataset_name == 'vcoco':
        from api.datasets import VCOCO as CGDataset
        image_size = 84
        padding_len = 8
    elif dataset_name == 'attr_obj':
        from api.datasets import ATTROBJ as CGDataset
        image_size = 84
        padding_len = 8
    elif dataset_name == 'action_obj':
        from api.datasets import ACTIONOBJ as CGDataset
        image_size = 84
        padding_len = 8
    else:
        raise ValueError('Non-supported Dataset.')

    # augmentations
    # Reference：https://github.com/Sha-Lab/FEAT
    if args.augment and phase == 'train':
        transforms_list = [
            transforms.RandomResizedCrop((image_size, image_size)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    else:
        transforms_list = [
            transforms.Resize((image_size+padding_len, image_size+padding_len)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]


    # pre-processing 
    if args.backbone == 'resnet12':
        transforms_list = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])

    else:
        transforms_list = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])


    # get dataset
    dataset = CGDataset(
        root=args.data_folder,
        meta_split=phase,
        transform=transforms_list,
        download=args.download,
    )

    # get sampler
    # if args.sampler == 'cg':
    #     from api.samplers import CGTaskSampler as TaskSampler
    # elif args.sampler == 'rcg':
    #     from api.samplers import RCGTaskSampler as TaskSampler
    # elif args.sampler == 'gcg':
    #     from api.samplers import GCGTaskSampler as TaskSampler
    # elif args.sampler == 'fsc':
    #     from api.samplers import FSCTaskSampler as TaskSampler
    # else:
    #     raise ValueError('Non-supported Sampler.')

    if args.sampler == 'fsc':
        from api.samplers import FSCTaskSampler as TaskSampler
    else:
        raise ValueError('Non-supported Sampler.')

    if phase == 'train':
        num_tasks = args.train_tasks
    elif phase == 'val':
        num_tasks = args.val_tasks
    else:
        num_tasks = args.test_tasks


    sampler = TaskSampler(
        labels=dataset.labels,
        label_set=dataset.label_set,
        num_tasks=num_tasks,
        num_ways=args.num_ways, 
        num_shots=args.num_shots, 
        # num_classes=args.num_classes, 
        num_query=args.num_query
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        num_workers=args.num_workers,
        batch_sampler=sampler,
        pin_memory=True
    )

    return dataloader


def get_fixed_dataloader(args, dataset_name, phase):

    if dataset_name == 'attr_obj':
        from api.datasets import ATTROBJ as CGDataset
        image_size = 84
        padding_len = 8
    elif dataset_name == 'action_obj':
        from api.datasets import ACTIONOBJ as CGDataset
        image_size = 84
        padding_len = 8
    else:
        raise ValueError('Non-supported Dataset.')

    # augmentations
    # Reference：https://github.com/Sha-Lab/FEAT
    if args.augment and phase == 'train':
        transforms_list = [
            transforms.RandomResizedCrop((image_size, image_size)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    else:
        transforms_list = [
            transforms.Resize((image_size+padding_len, image_size+padding_len)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]


    # pre-processing 
    if args.backbone == 'resnet12':
        transforms_list = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])

    else:
        transforms_list = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])


    # get dataset
    dataset = CGDataset(
        root=args.data_folder,
        meta_split=phase,
        transform=transforms_list,
        download=args.download,
    )

    # get sampler
    from api.samplers import Fixed_FSCTaskSampler as TaskSampler

    if phase == 'train':
        num_tasks = args.train_tasks
    elif phase == 'val':
        num_tasks = args.val_tasks
    else:
        num_tasks = args.test_tasks


    sampler = TaskSampler(
        labels=dataset.labels,
        label_set=dataset.label_set,
        num_tasks=num_tasks,
        num_ways=args.num_ways, 
        num_shots=args.num_shots, 
        # num_classes=args.num_classes, 
        num_query=args.num_query,
        num_seen=args.num_seen,
        num_unseen=args.num_unseen
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        num_workers=args.num_workers,
        batch_sampler=sampler,
        pin_memory=True
    )

    return dataloader


def label2vec(labels):
    index = 0
    label2index_dict = {}
    for label in set(labels):
        if label not in label2index_dict.keys():
            label2index_dict[label] = index
            index += 1
    vec = list()
    for label in labels:
        vec.append(label2index_dict[label])
    return torch.tensor(vec, dtype=torch.long)


def batch_process(args, batch):

    images = batch['images']
    labels_1_vec = label2vec(batch['labels_1'])
    labels_2_vec = label2vec(batch['labels_2'])

    assert len(images.shape) == 4    # [num_samples, c, h, w]

    if args.use_cuda:
        images = images.cuda(non_blocking=True)
        labels_1_vec = labels_1_vec.cuda(non_blocking=True)
        labels_2_vec = labels_2_vec.cuda(non_blocking=True)

    # get seen_num
    first_seen_count = args.num_shots+1
    for i_sample in range(labels_1_vec.shape[0]):
        if (labels_1_vec[i_sample] == labels_1_vec[0]) and (labels_2_vec[i_sample] == labels_2_vec[0]):
            first_seen_count -= 1
        if first_seen_count == 0:
            seen_num = i_sample // args.num_shots
            break
    unseen_num = ((labels_1_vec.shape[0] - seen_num * args.num_shots) // args.num_query) - seen_num

    # split data and labels into support and query
    support_inputs = images[:seen_num * args.num_shots].unsqueeze(0)
    query_inputs = images[seen_num * args.num_shots:].unsqueeze(0)
    support_labels_1_vec = labels_1_vec[:seen_num * args.num_shots].unsqueeze(0)
    query_labels_1_vec = labels_1_vec[seen_num * args.num_shots:].unsqueeze(0)
    support_labels_2_vec = labels_2_vec[:seen_num * args.num_shots].unsqueeze(0)
    query_labels_2_vec = labels_2_vec[seen_num * args.num_shots:].unsqueeze(0)

    return support_inputs, support_labels_1_vec, support_labels_2_vec, query_inputs, query_labels_1_vec, query_labels_2_vec, seen_num, unseen_num


def get_accuracies(logits_1, logits_2, labels_1, labels_2, seen_num, args):
    accuracies = dict()
    _, predictions_1 = torch.max(logits_1, dim=-1)
    _, predictions_2 = torch.max(logits_2, dim=-1)
        
    accuracies['seen_acc'] = torch.mean(
        (predictions_1[:seen_num * args.num_query].eq(labels_1[:seen_num * args.num_query]).float() + 
            predictions_2[:seen_num * args.num_query].eq(labels_2[:seen_num * args.num_query]).float()
        ).eq(2).float()
    )

    accuracies['unseen_acc'] = torch.mean(
        (predictions_1[seen_num * args.num_query:].eq(labels_1[seen_num * args.num_query:]).float() + 
            predictions_2[seen_num * args.num_query:].eq(labels_2[seen_num * args.num_query:]).float()
        ).eq(2).float()
    )

    accuracies['seen_prim1_acc'] = torch.mean(
        predictions_1[:seen_num * args.num_query].eq(labels_1[:seen_num * args.num_query]).float()
    )

    accuracies['seen_prim2_acc'] = torch.mean(
        predictions_2[:seen_num * args.num_query].eq(labels_2[:seen_num * args.num_query]).float()
    )

    accuracies['unseen_prim1_acc'] = torch.mean(
        predictions_1[seen_num * args.num_query:].eq(labels_1[seen_num * args.num_query:]).float()
    )

    accuracies['unseen_prim2_acc'] = torch.mean(
        predictions_2[seen_num * args.num_query:].eq(labels_2[seen_num * args.num_query:]).float()
    )

    return accuracies


def get_preds_and_accuracies(logits_1, logits_2, labels_1, labels_2, seen_num, args):
    accuracies = dict()
    _, predictions_1 = torch.max(logits_1, dim=-1)
    _, predictions_2 = torch.max(logits_2, dim=-1)
        
    accuracies['seen_acc'] = torch.mean(
        (predictions_1[:seen_num * args.num_query].eq(labels_1[:seen_num * args.num_query]).float() + 
            predictions_2[:seen_num * args.num_query].eq(labels_2[:seen_num * args.num_query]).float()
        ).eq(2).float()
    )

    accuracies['unseen_acc'] = torch.mean(
        (predictions_1[seen_num * args.num_query:].eq(labels_1[seen_num * args.num_query:]).float() + 
            predictions_2[seen_num * args.num_query:].eq(labels_2[seen_num * args.num_query:]).float()
        ).eq(2).float()
    )

    accuracies['seen_prim1_acc'] = torch.mean(
        predictions_1[:seen_num * args.num_query].eq(labels_1[:seen_num * args.num_query]).float()
    )

    accuracies['seen_prim2_acc'] = torch.mean(
        predictions_2[:seen_num * args.num_query].eq(labels_2[:seen_num * args.num_query]).float()
    )

    accuracies['unseen_prim1_acc'] = torch.mean(
        predictions_1[seen_num * args.num_query:].eq(labels_1[seen_num * args.num_query:]).float()
    )

    accuracies['unseen_prim2_acc'] = torch.mean(
        predictions_2[seen_num * args.num_query:].eq(labels_2[seen_num * args.num_query:]).float()
    )

    return accuracies, predictions_1, predictions_2, labels_1, labels_2