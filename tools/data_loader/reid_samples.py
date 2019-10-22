import sys
sys.path.append('..')

import os
import copy


def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files


class ReIDSamples:
    '''
    rgb_samples_train
    ir_samples_train
    rgb_samples_val
    ir_samples_val
    all_samples_test
    all_samples_train
    all_samples_val
    '''

    def __init__(self, dataset_path, reorder):

        self.dataset_path = dataset_path
        self.reorder = reorder
        # split rgb/ir cameras ids and train/val/test person ids
        self.rgb_camera_ids = [1,2,4,5]
        self.ir_camera_ids = [3,6]
        self._load_person_id_split()

        # load samples from dataset
        self._load_samples()

    def _load_samples(self):

        self.rgb_samples_train = []
        self.ir_samples_train = []
        self.rgb_samples_val = []
        self.ir_samples_val = []
        self.rgb_samples_test = []
        self.ir_samples_test = []
    
        # rgb cameras
        for cam_id in self.rgb_camera_ids:
            cam_path = os.path.join(self.dataset_path, 'cam{}'.format(cam_id))
            _, person_ids, _ = os_walk(cam_path)

            for person_id in person_ids:
                person_path = os.path.join(cam_path, person_id)
                _, _, image_names = os_walk(person_path)
                for image_name in image_names:
                    image_path = os.path.join(person_path, image_name)
                    sample = [image_path, int(person_id), cam_id, 0] # rgb
                    if int(person_id) in self.train_ids:
                        self.rgb_samples_train.append(sample)
                    elif int(person_id) in self.val_ids:
                        self.rgb_samples_val.append(sample)
                    elif int(person_id) in self.test_ids:
                        self.rgb_samples_test.append(sample)

        # ir cameras
        for cam_id in self.ir_camera_ids:
            cam_path = os.path.join(self.dataset_path, 'cam{}'.format(cam_id))
            _, person_ids, _ = os_walk(cam_path)
            for person_id in person_ids:
                person_path = os.path.join(cam_path, person_id)
                _, _, image_names = os_walk(person_path)
                for image_name in image_names:
                    image_path = os.path.join(person_path, image_name)
                    sample = [image_path, int(person_id), cam_id, 1] # ir
                    if int(person_id) in self.train_ids:
                        self.ir_samples_train.append(sample)
                    elif int(person_id) in self.val_ids:
                        self.ir_samples_val.append(sample)
                    elif int(person_id) in self.test_ids:
                        self.ir_samples_test.append(sample)

        print('Note: pids of rgb_train_set and ir_train_set are {} equal'.format(self._is_equal(self.rgb_samples_train, self.ir_samples_train, 1)))
        print('Note: pids of rgb_val_set and ir_val_set are {} equal'.format(self._is_equal(self.rgb_samples_val, self.ir_samples_val, 1)))


        self.rgb_samples_train = copy.deepcopy(self.rgb_samples_train + self.rgb_samples_val)
        self.ir_samples_train = copy.deepcopy(self.ir_samples_train + self.ir_samples_val)

        self.rgb_samples_all = copy.deepcopy(self.rgb_samples_train + self.rgb_samples_test)
        self.ir_samples_all = copy.deepcopy(self.ir_samples_train + self.ir_samples_test)

        if self.reorder:
            self.rgb_samples_train = self._reorder(self.rgb_samples_train, 1)
            self.ir_samples_train = self._reorder(self.ir_samples_train, 1)
            print('Note: Pids training and valation set are re-ordered separtely')

        print('rgb_samples_train', self._anaplsis_samples(self.rgb_samples_train))
        print('ir_samples_train', self._anaplsis_samples(self.ir_samples_train))
        print('rgb_samples_test', self._anaplsis_samples(self.rgb_samples_test))
        print('ir_samples_test', self._anaplsis_samples(self.ir_samples_test))
        print('rgb_samples_all', self._anaplsis_samples(self.rgb_samples_all))
        print('ir_samples_all', self._anaplsis_samples(self.ir_samples_all))


    def _load_person_id_split(self):

        train_ids_path = os.path.join(self.dataset_path, 'exp/train_id.txt')
        train_ids = open(train_ids_path).readline().replace('\n','').split(',')
        self.train_ids = list(map(int, train_ids))

        val_ids_path = os.path.join(self.dataset_path, 'exp/val_id.txt')
        val_ids = open(val_ids_path).readline().replace('\n', '').split(',')
        self.val_ids = list(map(int, val_ids))

        test_ids_path = os.path.join(self.dataset_path, 'exp/test_id.txt')
        test_ids = open(test_ids_path).readline().replace('\n','').split(',')
        self.test_ids = list(map(int, test_ids))


    def _reorder(self, samples, which_one):
        '''
        input [3, 5, 10, 9]
        output [0, 1, 3, 2]
        :param samples: [(), (), ...]
        :param which_one:  int
        :return:
        '''

        samples = copy.deepcopy(samples)

        ids = []
        for sample in samples:
            ids.append(sample[which_one])

        # delete repetitive elments and order
        ids = list(set(ids))
        ids.sort(reverse=False)
        # reorder
        for sample in samples:
            sample[which_one] = ids.index(sample[which_one])

        return samples


    def _is_equal(self, set1, set2, index):
        list1 = []
        list2 = []
        for sample1 in set1:
            list1.append(sample1[index])
        for sample2 in set2:
            list2.append(sample2[index])
        list1 = list(set(list1))
        list2 = list(set(list2))
        list1.sort()
        list2.sort()
        return list1 == list2

    def _anaplsis_samples(self, samples):

        samples = copy.deepcopy(samples)
        total_samples = len(samples)

        pids_list = []
        cids_list = []
        for sample in samples:
            _, pid, cid, _ = sample
            pids_list.append(pid)
            cids_list.append(cid)
        total_pids = len(set(pids_list))
        total_cids = len(set(cids_list))

        return total_samples, total_pids, total_cids
