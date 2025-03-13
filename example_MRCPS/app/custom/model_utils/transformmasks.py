import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.special import erfinv
import torch

def generate_tokenout_mask(img_size, lrratio=None):

    cutout_h = img_size[0] // 16
    cutout_w = img_size[1] // 16
    tokenratio = np.random.randint(8,17)
    tokennum = cutout_h * cutout_w // tokenratio

    mask = np.ones(img_size)
    for i in range(tokennum):
        h_start = np.random.randint(0, cutout_h) * cutout_h
        w_start = np.random.randint(0, cutout_w) * cutout_w
        mask[h_start:h_start+cutout_h, w_start:w_start+cutout_w] = 0
    if lrratio is not None:
        h_mid, w_mid = img_size[0]//2, img_size[1]//2
        h_res, w_res = h_mid//lrratio//2, w_mid//lrratio//2
        mask[h_mid-h_res:h_mid+h_res, w_mid-w_res:w_mid+w_res] = 1

    return mask.astype(float)

def generate_cutout_mask(img_size):

    cutout_area = img_size[0] * img_size[1] / 2

    w = np.random.randint(img_size[1] / 2, img_size[1] + 1)
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = np.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.astype(float)

def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N
