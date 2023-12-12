import torch
import numpy as np
import random


class CGTaskSampler():    # Compositional Generalization
    def __init__(self, labels, label_set, num_tasks, num_ways, num_shots, num_classes, num_query):
        '''
        Args:
            labels: The labels of the tuples of all samples
            label_set: The label set of the tuples
            num_tasks: Number of tasks in this stage
            num_ways: Number of classes of each kind of primitives per task 
                    = Number of classes of support tuples
            num_shots: Number of support examples per class
            num_classes: Number of classes of query tuples
            num_query: Number of query examples per class
        '''
        assert num_classes + num_ways <= num_ways * num_ways

        self.labels = labels
        self.label_set = label_set
        self.num_tasks = num_tasks
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_classes = num_classes
        self.num_query = num_query

        self.label2index = dict()
        
        for label in self.label_set:
            self.label2index[label] = list()

        for i in range(len(self.labels)):
            self.label2index[self.labels[i]].append(i)

        for label in self.label_set:
            self.label2index[label] = np.array(self.label2index[label])


    def __len__(self):
        return self.num_tasks
        

    def __iter__(self):
        for i_task in range(self.num_tasks):
            batch = []
            task_valid_flag = False
            while not task_valid_flag:
                # find n=num_ways support classes that the primitives only appear once
                sampled_labels_1 = list()
                sampled_labels_2 = list()
                support_labels = list()
                random_labels_index = torch.randperm(len(self.label_set))
                cur_index = 0
                while (len(support_labels) < self.num_ways) and (cur_index < len(random_labels_index)):
                    candidate_label = self.label_set[random_labels_index[cur_index]]
                    candidate_label_1, candidate_label_2 = candidate_label.split('_')
                    if (candidate_label_1 not in sampled_labels_1) and (candidate_label_2 not in sampled_labels_2):
                        sampled_labels_1.append(candidate_label_1)
                        sampled_labels_2.append(candidate_label_2)
                        support_labels.append(candidate_label)
                    cur_index += 1

                if not len(support_labels) == self.num_ways:
                    # raise RuntimeError('CG Sampling failed')
                    continue

                # make sure the task is valid, i.e., there are enough query classes
                candidate_query_labels = list()
                for candidate_label in self.label_set:
                    candidate_label_1, candidate_label_2 = candidate_label.split('_')
                    if (candidate_label_1 in sampled_labels_1) and (candidate_label_2 in sampled_labels_2) and (candidate_label not in support_labels):
                        candidate_query_labels.append(candidate_label)

                if len(candidate_query_labels) < self.num_classes:
                    continue
                else:
                    task_valid_flag = True

                # randomly select n=num_classes query classes
                query_labels_index = np.random.permutation(len(candidate_query_labels))[:self.num_classes]
                query_labels = np.array(candidate_query_labels)[query_labels_index]

                # find support data index
                for label in support_labels:
                    indices = self.label2index[label]
                    batch.append(indices[np.random.permutation(len(indices))[:self.num_shots]])

                # find query data index
                for label in query_labels:
                    indices = self.label2index[label]
                    batch.append(indices[np.random.permutation(len(indices))[:self.num_query]])

            batch = np.hstack(batch)

            yield batch


class RCGTaskSampler():    # Random Compositional Generalization: query compositions may contain support compositions
    def __init__(self, labels, label_set, num_tasks, num_ways, num_shots, num_classes, num_query):
        '''
        Args:
            labels: The labels of the tuples of all samples
            label_set: The label set of the tuples
            num_tasks: Number of tasks in this stage
            num_ways: Number of classes of each kind of primitives per task 
                    = Number of classes of support tuples
            num_shots: Number of support examples per class
            num_classes: Number of classes of query tuples
            num_query: Number of query examples per class
        '''
        assert num_classes <= num_ways * num_ways    # ? where difference lies

        self.labels = labels
        self.label_set = label_set
        self.num_tasks = num_tasks
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_classes = num_classes
        self.num_query = num_query

        self.label2index = dict()
        
        for label in self.label_set:
            self.label2index[label] = list()

        for i in range(len(self.labels)):
            self.label2index[self.labels[i]].append(i)

        for label in self.label_set:
            self.label2index[label] = np.array(self.label2index[label])


    def __len__(self):
        return self.num_tasks
        

    def __iter__(self):
        for i_task in range(self.num_tasks):
            batch = []
            task_valid_flag = False
            while not task_valid_flag:
                # find n=num_ways support classes that the primitives only appear once
                sampled_labels_1 = list()
                sampled_labels_2 = list()
                support_labels = list()
                random_labels_index = torch.randperm(len(self.label_set))
                cur_index = 0
                while (len(support_labels) < self.num_ways) and (cur_index < len(random_labels_index)):
                    candidate_label = self.label_set[random_labels_index[cur_index]]
                    candidate_label_1, candidate_label_2 = candidate_label.split('_')
                    if (candidate_label_1 not in sampled_labels_1) and (candidate_label_2 not in sampled_labels_2):
                        sampled_labels_1.append(candidate_label_1)
                        sampled_labels_2.append(candidate_label_2)
                        support_labels.append(candidate_label)
                    cur_index += 1

                if not len(support_labels) == self.num_ways:
                    # raise RuntimeError('RCG Sampling failed')
                    continue

                # make sure the task is valid, i.e., there are enough query classes
                candidate_query_labels = list()
                for candidate_label in self.label_set:
                    candidate_label_1, candidate_label_2 = candidate_label.split('_')
                    if (candidate_label_1 in sampled_labels_1) and (candidate_label_2 in sampled_labels_2):
                        candidate_query_labels.append(candidate_label)

                if len(candidate_query_labels) < self.num_classes:
                    continue
                else:
                    task_valid_flag = True

                # randomly select n=num_classes query classes
                query_labels_index = np.random.permutation(len(candidate_query_labels))[:self.num_classes]
                query_labels = np.array(candidate_query_labels)[query_labels_index]

                # find support data index
                for label in support_labels:
                    indices = self.label2index[label]
                    batch.append(indices[np.random.permutation(len(indices))[:self.num_shots]])

                # find query data index
                for label in query_labels:
                    indices = self.label2index[label]
                    batch.append(indices[np.random.permutation(len(indices))[:self.num_query]])

            batch = np.hstack(batch)

            yield batch


class GCGTaskSampler():    # Generalized Compositional Generalization: query compositions must contain support compositions and unseen compositions
    def __init__(self, labels, label_set, num_tasks, num_ways, num_shots, num_classes, num_query):
        '''
        Args:
            labels: The labels of the tuples of all samples
            label_set: The label set of the tuples
            num_tasks: Number of tasks in this stage
            num_ways: Number of classes of each kind of primitives per task 
                    = Number of classes of support tuples
            num_shots: Number of support examples per class
            num_classes: Number of classes of query tuples
            num_query: Number of query examples per class
        '''
        assert num_classes <= num_ways * num_ways    # ? where difference lies
        assert num_ways <= num_classes

        self.labels = labels
        self.label_set = label_set
        self.num_tasks = num_tasks
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_classes = num_classes
        self.num_query = num_query
        self.label2index = dict()
        
        for label in self.label_set:
            self.label2index[label] = list()

        for i in range(len(self.labels)):
            self.label2index[self.labels[i]].append(i)

        for label in self.label_set:
            self.label2index[label] = np.array(self.label2index[label])


    def __len__(self):
        return self.num_tasks
        

    def __iter__(self):
        for i_task in range(self.num_tasks):
            batch = []
            task_valid_flag = False
            while not task_valid_flag:
                # find n=num_ways support classes that the primitives only appear once
                sampled_labels_1 = list()
                sampled_labels_2 = list()
                support_labels = list()
                random_labels_index = torch.randperm(len(self.label_set))
                cur_index = 0
                while (len(support_labels) < self.num_ways) and (cur_index < len(random_labels_index)):
                    candidate_label = self.label_set[random_labels_index[cur_index]]
                    candidate_label_1, candidate_label_2 = candidate_label.split('_')
                    if (candidate_label_1 not in sampled_labels_1) and (candidate_label_2 not in sampled_labels_2):
                        sampled_labels_1.append(candidate_label_1)
                        sampled_labels_2.append(candidate_label_2)
                        support_labels.append(candidate_label)
                    cur_index += 1

                if not len(support_labels) == self.num_ways:
                    # raise RuntimeError('GCG Sampling failed')
                    continue

                # make sure the task is valid, i.e., there are enough query classes
                candidate_query_labels = list()
                for candidate_label in self.label_set:
                    candidate_label_1, candidate_label_2 = candidate_label.split('_')
                    if (candidate_label_1 in sampled_labels_1) and (candidate_label_2 in sampled_labels_2) and (candidate_label not in support_labels):
                        candidate_query_labels.append(candidate_label)

                if len(candidate_query_labels) < self.num_classes - self.num_ways:
                    continue
                else:
                    task_valid_flag = True

                # randomly select n=(num_classes-num_ways) query classes, and add support classes into query classes
                sampled_num_classes = self.num_classes - self.num_ways
                query_labels_index = np.random.permutation(len(candidate_query_labels))[:sampled_num_classes]

                # make sure that support and query samples are not overlap
                query_labels = np.array(candidate_query_labels)[query_labels_index]

                seen_classes_query_samples = list()
                for label in support_labels:
                    indices = self.label2index[label]
                    permuted_index_of_index = np.random.permutation(len(indices))
                    batch.append(indices[permuted_index_of_index[:self.num_shots]])
                    seen_classes_query_samples.append(indices[permuted_index_of_index[self.num_shots:self.num_shots+self.num_query]])

                batch = batch + seen_classes_query_samples

                # find query data index
                for label in query_labels:
                    indices = self.label2index[label]
                    batch.append(indices[np.random.permutation(len(indices))[:self.num_query]])

            # The order of the data in the batch: [support samples of seen compositions, query samples of seen compositions, query samples of unseen compositions]
            batch = np.hstack(batch)

            yield batch


class FSCTaskSampler():    # seen >= 6，unseen >= 1
    def __init__(self, labels, label_set, num_tasks, num_ways, num_shots, num_query):
        '''
        Args:
            labels: The labels of the tuples of all samples
            label_set: The label set of the tuples
            num_tasks: Number of tasks in this stage
            num_ways: Number of classes of each kind of primitives per task 
            num_shots: Number of support examples per class
            num_query: Number of query examples per class
        '''
        # assert num_classes <= num_ways * num_ways    # ? where difference lies
        # assert num_ways <= num_classes

        self.labels = labels
        self.label_set = label_set
        self.num_tasks = num_tasks
        self.num_ways = num_ways
        self.num_shots = num_shots
        # self.num_classes = num_classes
        self.num_query = num_query
        self.label2index = dict()
        
        for label in self.label_set:
            self.label2index[label] = list()

        for i in range(len(self.labels)):
            self.label2index[self.labels[i]].append(i)

        for label in self.label_set:
            if len(self.label2index[label]) < num_shots+num_query:
                print('label:', label)
            assert len(self.label2index[label]) >= num_shots+num_query
            self.label2index[label] = np.array(self.label2index[label])


    def __len__(self):
        return self.num_tasks
        

    def __iter__(self):
        for i_task in range(self.num_tasks):
            batch = []
            task_valid_flag = False
            while not task_valid_flag:
                # find num_ways support classes that the primitives only appear once
                sampled_labels_1 = list()
                sampled_labels_2 = list()
                support_labels = list()
                random_labels_index = torch.randperm(len(self.label_set))
                cur_index = 0
                while (len(sampled_labels_1) < self.num_ways) and (cur_index < len(random_labels_index)):
                    candidate_label = self.label_set[random_labels_index[cur_index]]
                    candidate_label_1, candidate_label_2 = candidate_label.split('_')
                    if (candidate_label_1 not in sampled_labels_1) and (candidate_label_2 not in sampled_labels_2):
                        sampled_labels_1.append(candidate_label_1)
                        sampled_labels_2.append(candidate_label_2)
                        support_labels.append(candidate_label)
                    cur_index += 1

                if not len(sampled_labels_1) == self.num_ways:
                    # raise RuntimeError('GCG Sampling failed')
                    continue

                # find candidate_labels that can be seen or unseen compositions
                candidate_labels = list()
                for candidate_label in self.label_set:
                    candidate_label_1, candidate_label_2 = candidate_label.split('_')
                    if (candidate_label_1 in sampled_labels_1) and (candidate_label_2 in sampled_labels_2) and (candidate_label not in support_labels):
                        candidate_labels.append(candidate_label)

                # make sure the task is valid, i.e., there are enough query classes
                if len(candidate_labels) >= 2:
                    task_valid_flag = True
                else:
                    continue
                # randomly sampling the number of seen compositions
                seen_num = np.random.randint(self.num_ways+1, self.num_ways+len(candidate_labels))
                # unseen_num = self.num_ways + len(candidate_labels) - seen_num

                # sample the remaining seen_num-num_ways support classes / seen compositions
                random_candidate_labels_index = np.random.permutation(len(candidate_labels))
                for i_label in random_candidate_labels_index[:seen_num-self.num_ways].tolist():
                    support_labels.append(candidate_labels[i_label])
                query_labels = list()
                for i_label in random_candidate_labels_index[seen_num-self.num_ways:].tolist():
                    query_labels.append(candidate_labels[i_label])

                # sample samples for seen compositions
                seen_classes_query_samples = list()
                for label in support_labels:
                    indices = self.label2index[label]
                    permuted_index_of_index = np.random.permutation(len(indices))
                    batch.append(indices[permuted_index_of_index[:self.num_shots]])
                    seen_classes_query_samples.append(indices[permuted_index_of_index[self.num_shots:self.num_shots+self.num_query]])

                batch = batch + seen_classes_query_samples

                # sample query samples for unseen compositions
                for label in query_labels:
                    indices = self.label2index[label]
                    batch.append(indices[np.random.permutation(len(indices))[:self.num_query]])

            # The order of the data in the batch: [support samples of seen compositions, query samples of seen compositions, query samples of unseen compositions]
            batch = np.hstack(batch)

            yield batch


class Fixed_FSCTaskSampler():    # based on FSCTaskSampler, fix the number of seen and unseen
    def __init__(self, labels, label_set, num_tasks, num_ways, num_shots, num_query, num_seen, num_unseen=-1, max_iter=500):
        '''
        Args:
            labels: The labels of the tuples of all samples
            label_set: The label set of the tuples
            num_tasks: Number of tasks in this stage
            num_ways: Number of classes of each kind of primitives per task 
            num_shots: Number of support examples per class
            num_query: Number of query examples per class

            num_seen: Number of seen compositions
            num_unseen: Number of unseen compositions
            max_iter: Number of max iters for searching the next episode
        '''
        # assert num_classes <= num_ways * num_ways    # ? where difference lies
        # assert num_ways <= num_classes

        self.labels = labels
        self.label_set = label_set
        self.num_tasks = num_tasks
        self.num_ways = num_ways
        self.num_shots = num_shots
        # self.num_classes = num_classes
        self.num_query = num_query
        self.num_seen = num_seen
        self.num_unseen = num_unseen
        self.max_iter = max_iter
        self.label2index = dict()

        assert num_seen > num_ways
        
        for label in self.label_set:
            self.label2index[label] = list()

        for i in range(len(self.labels)):
            self.label2index[self.labels[i]].append(i)

        for label in self.label_set:
            if len(self.label2index[label]) < num_shots+num_query:
                print('label:', label)
            assert len(self.label2index[label]) >= num_shots+num_query
            self.label2index[label] = np.array(self.label2index[label])


    def __len__(self):
        return self.num_tasks
        

    def __iter__(self):
        for i_task in range(self.num_tasks):
            batch = []
            task_valid_flag = False
            iter_counter = 0
            while (not task_valid_flag) and (iter_counter < self.max_iter):
                iter_counter += 1
                # find num_ways support classes that the primitives only appear once
                sampled_labels_1 = list()
                sampled_labels_2 = list()
                support_labels = list()
                random_labels_index = torch.randperm(len(self.label_set))
                cur_index = 0
                while (len(sampled_labels_1) < self.num_ways) and (cur_index < len(random_labels_index)):
                    candidate_label = self.label_set[random_labels_index[cur_index]]
                    candidate_label_1, candidate_label_2 = candidate_label.split('_')
                    if (candidate_label_1 not in sampled_labels_1) and (candidate_label_2 not in sampled_labels_2):
                        sampled_labels_1.append(candidate_label_1)
                        sampled_labels_2.append(candidate_label_2)
                        support_labels.append(candidate_label)
                    cur_index += 1

                if not len(sampled_labels_1) == self.num_ways:
                    # raise RuntimeError('GCG Sampling failed')
                    continue

                # find candidate_labels that can be seen or unseen compositions
                candidate_labels = list()
                for candidate_label in self.label_set:
                    candidate_label_1, candidate_label_2 = candidate_label.split('_')
                    if (candidate_label_1 in sampled_labels_1) and (candidate_label_2 in sampled_labels_2) and (candidate_label not in support_labels):
                        candidate_labels.append(candidate_label)

                # make sure the task is valid
                min_num_unseen = 1 if self.num_unseen == -1 else self.num_unseen
                if len(candidate_labels) >= self.num_seen - self.num_ways + min_num_unseen:
                    task_valid_flag = True
                else:
                    continue

                # sample the remaining seen_num-num_ways support classes / seen compositions
                random_candidate_labels_index = np.random.permutation(len(candidate_labels))

                for i_label in random_candidate_labels_index[:self.num_seen-self.num_ways].tolist():
                    support_labels.append(candidate_labels[i_label])
                query_labels = list()
                if self.num_unseen == -1:    # num_unseen is not fixed
                    for i_label in random_candidate_labels_index[self.num_seen-self.num_ways:].tolist():
                        query_labels.append(candidate_labels[i_label])
                else:
                    for i_label in random_candidate_labels_index[self.num_seen-self.num_ways:self.num_seen-self.num_ways+self.num_unseen].tolist():
                        query_labels.append(candidate_labels[i_label])

                # sample samples for seen compositions
                seen_classes_query_samples = list()
                for label in support_labels:
                    indices = self.label2index[label]
                    permuted_index_of_index = np.random.permutation(len(indices))
                    batch.append(indices[permuted_index_of_index[:self.num_shots]])
                    seen_classes_query_samples.append(indices[permuted_index_of_index[self.num_shots:self.num_shots+self.num_query]])

                batch = batch + seen_classes_query_samples

                # sample query samples for unseen compositions
                for label in query_labels:
                    indices = self.label2index[label]
                    batch.append(indices[np.random.permutation(len(indices))[:self.num_query]])

            if iter_counter >= self.max_iter:
                raise RuntimeError('Cannot find valid episodes for {} seen {} unseen!'.format(str(self.num_seen), str(self.num_unseen)))

            # The order of the data in the batch: [support samples of seen compositions, query samples of seen compositions, query samples of unseen compositions]
            batch = np.hstack(batch)

            yield batch