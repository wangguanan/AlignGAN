'''
This evaluation code is modified from Ancong Wu
'''


import scipy.io as sio
import numpy as np
import argparse
import os
import pdb
import multiprocessing
import time

args = None


class DataCam(object):
    def __init__(self, data=None):
        self.data = data if data is not None else {}

    def select_id(self, ids):
        data = dict([(id_, self.data[id_]) for id_ in ids if id_ in self.data])
        ret = DataCam(data)
        return ret

    def to_list(self):
        features = []
        labels = []
        for id_ in self.data:
            features.append(self.data[id_])
            labels.extend([id_] * len(self.data[id_]))
        features = np.concatenate(features)
        labels = np.array(labels)
        return features, labels

    def select_gallery(self, rand_perm, ntime, nshot):
        ret = []
        for i in range(ntime):
            data = {}
            for id_ in self.data:
                f = self.data[id_]
                ixs = rand_perm[id_][i][:nshot]
                data[id_] = f[ixs]
            ret.append(DataCam(data))
        return ret

    @classmethod
    def merge(cls, datacams):
        data = {}
        for datacam in datacams:
            for id_ in datacam.data:
                if id_ not in data:
                    data[id_] = []
                data[id_].append(datacam.data[id_])
        for id_ in data:
            data[id_] = np.concatenate(data[id_])
        ret = DataCam(data)
        return ret

    @classmethod
    def from_file(cls, filename):
        mat = sio.loadmat(filename)['feature']
        data = dict([(i, f[0]) for i, f in enumerate(mat) if f[0].shape[1] != 0])
        ret = DataCam(data)
        return ret


class Args(object):
    feature_dir = './'
    label_dir = './'
    mode = 'all'
    shot = 'single'
    times = 10
    proc_mode = 'multi'


def calc(gs_datacam, p_datacam, use_multi_proc):
    '''
    -gs_datacam:(ntimes)
    -p_datacam:(1)
    -use_multi_proc:(1)
    -ret_cmc:(n_ranks)
    -ret_map:(ntimes)
    '''

    ntimes = len(gs_datacam)
    p_args = [(g_datacam, p_datacam) for g_datacam in gs_datacam]

    if use_multi_proc:
        pool = multiprocessing.Pool(ntimes)
        ret = pool.map(calc_one, p_args)
        pool.close()
        pool.join()
    else:
        ret = list(map(calc_one, p_args))

    firsthits, apss = zip(*ret)
    firsthit = np.sum(np.array(firsthits), axis=0)
    aps = np.concatenate(apss)
    return firsthit, aps


def calc_one(args):
    '''
    -ret_firsthit:(nranks)
    -ret_aps:(p_record)
    '''
    g_datacam, p_datacam = args
    g_features, g_labels = g_datacam.to_list()
    p_features, p_labels = p_datacam.to_list()

    nid = len(g_datacam.data)
    nranks = len(g_features)

    firsthit = np.zeros(nid, dtype=np.int32)
    aps = []
    for i in range(len(p_features)):
        p_feature = p_features[i]
        p_label = p_labels[i]
        ranks, minrank = calc_rank(g_features, p_feature, g_labels, p_label)
        firsthit[minrank] += 1
        ap = np.mean(np.arange(1, len(ranks) + 1, dtype=np.float32) / (ranks + 1))
        aps.append(ap)
    aps = np.array(aps)
    return firsthit, aps


def calc_rank(g_features, p_feature, g_labels, p_label):
    '''
    -g_features:(n_record,n_feature)
    -p_feature:(n_feature)
    -g_labels:(n_record)
    -p_label:(1)
    -ret_ranks:(n_record_p_label_of_g)
    -ret_minranks:(1)
    '''
    # calc dist
    dists = calc_dist(g_features, p_feature)
    # calc ranks
    ixs = np.argsort(dists)
    ixs = np.flip(ixs, axis=0)
    g_plabel = np.flatnonzero(g_labels == p_label)
    beg_ix = np.min(g_plabel)
    end_ix = np.max(g_plabel) + 1
    ranks = np.flatnonzero((beg_ix <= ixs) & (ixs < end_ix))
    # calc minrank
    maxid = np.max(g_labels)
    tmp = np.zeros(maxid + 1, dtype=np.int32)
    cnt = 0
    for ix in ixs:
        id_ = g_labels[ix]
        if id_ == p_label:
            break
        if tmp[id_] == 0:
            tmp[id_] = 1
            cnt += 1
    minrank = cnt
    return ranks, minrank


def calc_dist(f1, f2):
    '''
    -f1:(n_feature) or (n_record,n_feature)
    -f2:(n_feature)
    -ret:(1) or (n_record)
    '''
    l1 = np.sqrt(np.sum(f1 ** 2, axis=-1))
    l2 = np.sqrt(np.sum(f2 ** 2, axis=-1))
    inner = np.sum(f1 * f2, axis=-1)
    return inner / (l1 * l2)


def evaluate_sysymm01(feature_dir, mode, shot):
    '''
    args:
        -args.data_dir:str, dir of data
        -args.mode:str, 'all'|'indoor'
        -args.shot:str, 'single'|'multi'
        -args.times:int, times of calc the ans
        -args.proc_mode:str, decide if use multi-process to speed up, 'single'|'multi'
    ret:[CMC,mAP]
        -CMC:np.array,shape=[nid]
        -mAP:float

    note:the names of data files are hardcoded,
         including 'feature_cam*.mat','test_id.mat','test_id_indoor.mat','rand_perm_cam.mat'
    '''

    args = Args()
    args.feature_dir = feature_dir
    args.label_dir = os.path.join(os.getcwd(), 'tools/evaluation/sysu_mm01_python/data_split/')
    args.mode = mode
    args.shot = shot
    args.times = 10
    args.proc_mode = 'multi'


    # io-read
    datacams = [DataCam.from_file(os.path.join(args.feature_dir, 'feature_cam{}.mat'.format(i + 1))) for i in range(6)]
    if args.mode == 'indoor':
        testid = sio.loadmat(os.path.join(args.label_dir, 'test_id_indoor.mat'))['id'][0]
    elif args.mode == 'all':
        testid = sio.loadmat(os.path.join(args.label_dir, 'test_id.mat'))['id'][0]
    else:
        raise Exception('invaild mode:{}'.format(args.mode))
    testid -= 1
    rand_perms = sio.loadmat(os.path.join(args.label_dir, 'rand_perm_cam.mat'))['rand_perm_cam']
    rand_perms = [dict([(i, f[0] - 1) for i, f in enumerate(r[0])]) for r in rand_perms]  # [cam]{id}[10,n_record]

    # decide nshot
    if args.shot == 'single':
        nshot = 1
    elif args.shot == 'multi':
        nshot = 10
    else:
        raise Exception('invaild shot:{}'.format(args.shot))
    # decide use_multi_proc
    if args.proc_mode == 'single':
        use_multi_proc = False
    elif args.proc_mode == 'multi':
        use_multi_proc = True
    else:
        raise Exception('invaild proc_mode:{}'.format(args.proc_mode))

    # select id
    datacams = [datacam.select_id(testid) for datacam in datacams]  # [cam]
    # select gallery
    gs_datacams = [datacam.select_gallery(rand_perms[i], args.times, nshot) for i, datacam in enumerate(datacams)]
    gs_datacams = list(zip(*gs_datacams))  # [ntimes][cam]

    if args.mode == 'all':
        g1s_datacam = [DataCam.merge([g_datacams[i - 1] for i in [1, 4, 5]]) for g_datacams in gs_datacams]  # [ntimes]
        p1_datacam = datacams[3 - 1]  # [1]
        g2s_datacam = [DataCam.merge([g_datacams[i - 1] for i in [1, 2, 4, 5]]) for g_datacams in gs_datacams]
        p2_datacam = datacams[6 - 1]

        fh1, aps1 = calc(g1s_datacam, p1_datacam, use_multi_proc)
        fh2, aps2 = calc(g2s_datacam, p2_datacam, use_multi_proc)

        fh = fh1 + fh2
        aps = np.concatenate((aps1, aps2))

        CMC = np.cumsum(fh)
        CMC = CMC / CMC[-1]
        mAP = np.mean(aps)
    elif args.mode == 'indoor':
        g1s_datacam = [DataCam.merge([g_datacams[i - 1] for i in [1]]) for g_datacams in gs_datacams]
        p1_datacam = datacams[3 - 1]
        g2s_datacam = [DataCam.merge([g_datacams[i - 1] for i in [1, 2]]) for g_datacams in gs_datacams]
        p2_datacam = datacams[6 - 1]

        fh1, aps1 = calc(g1s_datacam, p1_datacam, use_multi_proc)
        fh2, aps2 = calc(g2s_datacam, p2_datacam, use_multi_proc)

        fh = fh1 + fh2
        aps = np.concatenate((aps1, aps2))

        CMC = np.cumsum(fh)
        CMC = CMC / CMC[-1]
        mAP = np.mean(aps)
    else:
        raise Exception('invaild mode:{}'.format(args.mode))

    return CMC, mAP


if __name__ == '__main__':
    args = Args()
    args.data_dir = '/home/share/jianheng/rgbir'
    args.mode = 'all'
    args.shot = 'single'
    args.times = 10
    args.proc_mode = 'multi'

    CMC, mAP = main(args)


    print(CMC[:20])
    print(mAP)
