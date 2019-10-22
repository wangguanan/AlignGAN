import torch

class BestMeter:

    def __init__(self, better='larger'):
        self.better = better
        self.reset()

    def reset(self):
        if self.better == 'larger':
            self.val = -999999
        elif self.better == 'smaller':
            self.val = 999999
        self.flag = False

    def update(self, x):
        if self.better == 'larger':
            if x >= self.val:
                self.val = x
                self.flag = True
        elif self.better == 'smaller':
            if x <= self.val:
                self.val = x
                self.flag = True

    def get_flag(self):
        tmp = self.flag
        self.flag = False
        return tmp

    def get_val(self):
        return self.val


class AverageMeter:

    def __init__(self, neglect_value=None):
        self.reset()
        self.neglect_value = neglect_value

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n):
        if self.neglect_value is None or self.neglect_value not in val:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    def get_val(self):
        return self.avg

    def get_val_numpy(self):
        return self.avg.data.cpu().numpy()


class CatMeter:
    '''
    Concatenate Meter for torch.Tensor
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None

    def update(self, val):
        if self.val is None:
            self.val = val
        else:
            self.val = torch.cat([self.val, val], dim=0)
    def get_val(self):
        return self.val

    def get_val_numpy(self):
        return self.val.data.cpu().numpy()


class AppendMeter:
    '''
    Concatenate Meter for torch.Tensor
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None

    def update(self, val):
        if self.val is None:
            self.val = [val]
        else:
            self.val.append(val)

    def get_val(self):
        return self.val


class CountMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}

    def update(self, val):
        if val not in list(self.val.keys()):
            self.val[val] = 1
        else:
            self.val[val] += 1

    def get_val(self, val):
        if val not in list(self.val.keys()):
            return 0
        else:
            return self.val[val]

