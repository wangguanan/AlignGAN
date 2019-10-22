import torch.utils.data as data
import torchvision.transforms as transforms
import copy
import numpy as np
from .reid_samples import ReIDSamples
from .reid_dataset import ReIDDataSet
from .loader import UniformSampler, IterLoader, Seeds


class Loaders:

    def __init__(self, config, transform_train, transform_test):

        #  dataset configuration
        self.dataset_path = config.dataset_path

        # sample configuration
        self.p_gan = config.p_gan
        self.k_gan = config.k_gan
        self.p_ide = config.p_ide
        self.k_ide = config.k_ide

        # transforms
        self.transform_train = transform_train
        self.transform_test = transform_test

        # init loaders
        self._init_train_loaders()


    def _init_train_loaders(self):

        all_samples = ReIDSamples(self.dataset_path, True)

        # init datasets
        rgb_train_dataset = ReIDDataSet(all_samples.rgb_samples_train, self.transform_train)
        ir_train_dataset = ReIDDataSet(all_samples.ir_samples_train, self.transform_train)
        rgb_test_dataset = ReIDDataSet(all_samples.rgb_samples_test, self.transform_test)
        ir_test_dataset = ReIDDataSet(all_samples.ir_samples_test, self.transform_test)
        rgb_all_dataset = ReIDDataSet(all_samples.rgb_samples_all, self.transform_test)
        ir_all_dataset = ReIDDataSet(all_samples.ir_samples_all, self.transform_test)

        # init loaders
        seeds = Seeds(np.random.randint(0, 1e8, 9999))

        self.rgb_train_loader_gan = data.DataLoader(copy.deepcopy(rgb_train_dataset), self.p_gan * self.k_gan, shuffle=False,
                                                sampler=UniformSampler(rgb_train_dataset, self.k_gan, copy.copy(seeds)),
                                                num_workers=4, drop_last=True)
        self.ir_train_loader_gan = data.DataLoader(copy.deepcopy(ir_train_dataset), self.p_gan * self.k_gan, shuffle=False,
                                                sampler=UniformSampler(ir_train_dataset, self.k_gan, copy.copy(seeds)),
                                                num_workers=4, drop_last=True)

        self.rgb_train_loader_ide = data.DataLoader(copy.deepcopy(rgb_train_dataset), self.p_ide * self.k_ide, shuffle=False,
                                                sampler=UniformSampler(rgb_train_dataset, self.k_ide, copy.copy(seeds)),
                                                num_workers=8, drop_last=True)
        self.ir_train_loader_ide = data.DataLoader(copy.deepcopy(ir_train_dataset), self.p_ide * self.k_ide, shuffle=False,
                                                sampler=UniformSampler(ir_train_dataset, self.k_ide, copy.copy(seeds)),
                                                num_workers=8, drop_last=True)

        self.rgb_test_loader = data.DataLoader(rgb_test_dataset, 128, shuffle=False, num_workers=8, drop_last=False)
        self.ir_test_loader = data.DataLoader(ir_test_dataset, 128, shuffle=False, num_workers=8, drop_last=False)

        self.rgb_all_loader = data.DataLoader(rgb_all_dataset, 128, shuffle=False, num_workers=8, drop_last=False)
        self.ir_all_loader = data.DataLoader(ir_all_dataset, 128, shuffle=False, num_workers=8, drop_last=False)

        # init iters
        self.rgb_train_iter_gan = IterLoader(self.rgb_train_loader_gan)
        self.ir_train_iter_gan = IterLoader(self.ir_train_loader_gan)
        self.rgb_train_iter_ide = IterLoader(self.rgb_train_loader_ide)
        self.ir_train_iter_ide = IterLoader(self.ir_train_loader_ide)

