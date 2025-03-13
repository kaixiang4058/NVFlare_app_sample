import numpy as np
import kornia
import torch
import torch.nn as nn

def colorJitter(colorJitter, Threshold=0.4, data = None, lrdata = None, target = None, 
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, issymetric = True):
    """
    Apply colorJitter to tensor object
    Args:
        colorJitter: the probability of applying
        data, target: batch of data
        s: the value of augmentation
    """

    if not (data is None) and data.shape[1]==3 and colorJitter > Threshold:
        aug = kornia.augmentation.ColorJitter(
            brightness=brightness,contrast=contrast,saturation=saturation,hue=hue)
        data = aug(data)
        if not (lrdata is None) and lrdata.shape[1]==3:
            lrdata = aug(lrdata, params=aug._params) if issymetric else aug(lrdata)

    return data, lrdata, target

def gaussian_blur(blur, data = None, target = None):
    if not (data is None):
        if data.shape[1]==3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15,1.15)
                kernel_size_y = int(np.floor(np.ceil(0.1 * data.shape[2]) - 0.5 + np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(np.floor(np.ceil(0.1 * data.shape[3]) - 0.5 + np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(kornia.filters.GaussianBlur2d(kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target

def flip(flip, data = None, lrdata = None, target = None):
    #Flip
    if flip % 4 == 1:
        dim = (3,)
    elif flip % 4 == 2:
        dim = (2,)
    elif flip % 4 == 3:
        dim = (2,3)
    if flip % 4 != 0:
        if not (data is None): data = torch.flip(data,dim)
        if not (lrdata is None): lrdata = torch.flip(lrdata,dim)
        if not (target is None): target = torch.flip(target,dim)
    if flip >= 4:
        if not (data is None): data = torch.transpose(data, 2, 3)
        if not (lrdata is None): lrdata = torch.transpose(lrdata, 2, 3)
        if not (target is None): target = torch.transpose(target, 2, 3)
    
    return data, lrdata, target

def cowMix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask, data = torch.broadcast_tensors(mask, data)
        stackedMask = stackedMask.clone()
        stackedMask[1::2]=1-stackedMask[1::2]
        data = (stackedMask*torch.cat((data[::2],data[::2]))+(1-stackedMask)*torch.cat((data[1::2],data[1::2]))).float()
    if not (target is None):
        stackedMask, target = torch.broadcast_tensors(mask, target)
        stackedMask = stackedMask.clone()
        stackedMask[1::2]=1-stackedMask[1::2]
        target = (stackedMask*torch.cat((target[::2],target[::2]))+(1-stackedMask)*torch.cat((target[1::2],target[1::2]))).float()
    return data, target

def mix(mask, data = None, target = None):
    #Mix
    if not (mask is None):
        if not (data is None):
            if mask.shape[0] == data.shape[0]:
                data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])
            elif mask.shape[0] == data.shape[0] / 2:
                data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))]),
                                torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))])))
        if not (target is None):
            target = torch.cat([(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
    return data, target

def cutout(mask, data = None, target = None):
    #Mix
    if not (mask is None):
        if not (data is None):
            data = torch.cat([(mask[i] * data[i]).unsqueeze(0) for i in range(data.shape[0])])
        if not (target is None):
            target = torch.cat([(mask[i] * target[i]).unsqueeze(0) for i in range(target.shape[0])]).long()

    return data, target

def oneMix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0*data[0]+(1-stackedMask0)*data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0*target[0]+(1-stackedMask0)*target[1]).unsqueeze(0)
    return data, target
