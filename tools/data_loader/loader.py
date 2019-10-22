import sys
sys.path.append('..')

import random

import torch.utils.data as data
import numpy as np


class UniformSampler(data.sampler.Sampler):

    def __init__(self, dataset, k, random_seeds):

        self.dataset = dataset
        self.k = k
        self.random_seeds = random_seeds

        self.samples = self.dataset.samples
        self._process()

        self.sample_list = self._generate_list()


    def __iter__(self):
        self.sample_list = self._generate_list()
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _process(self):
        pids, cids = [], []
        for sample in self.samples:
            _, pid, cid, _ = sample
            pids.append(pid)
            cids.append(cid)

        self.pids = np.array(pids)
        self.cids = np.array(cids)

    def _generate_list(self):

        index_list = []
        pids = list(set(self.pids))
        pids.sort()

        seed = self.random_seeds.next_one()
        random.seed(seed)
        random.shuffle(pids)

        for pid in pids:
            # find all indexes of the person of pid
            index_of_pid = np.where(self.pids == pid)[0]
            # randomly sample k images from the pid
            if len(index_of_pid) >= self.k:
                index_list.extend(np.random.choice(index_of_pid, self.k, replace=False).tolist())
            else:
                index_list.extend(np.random.choice(index_of_pid, self.k, replace=True).tolist())

        return index_list


class IterLoader:

    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


class Seeds:

    def __init__(self, seeds):
        self.index = -1
        self.seeds = seeds

    def next_one(self):
        self.index += 1
        if self.index > len(self.seeds)-1:
            self.index = 0
        return self.seeds[self.index]

