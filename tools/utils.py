import os
import time

import torch
import numpy as np

def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files


def time_now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Successfully make dirs: {}'.format(dir))
    else:
        print('Existed dirs: {}'.format(dir))


def label2onehot(labels, dim, cuda):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    if cuda:
        out = out.to(torch.device('cuda'))
    return out


def analyze_names_and_meter(loss_names, loss_meter):

    result = ''
    for i in range(len(loss_names)):

        loss_name = loss_names[i]
        loss_value = loss_meter[i]

        result += loss_name
        result += ': '
        result += str(loss_value)
        result += ';  '

    return result
