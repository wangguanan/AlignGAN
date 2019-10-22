import torch

def fliplr(img, device=None):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    if device is not None:
        inv_idx = inv_idx.to(device)
    img_flip = img.index_select(3,inv_idx)
    return img_flip