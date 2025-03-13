import os
import pyvips
import numpy as np
import torch
import torch.nn.functional as F
import random
from utils.dataset import SemiSegDataset
from math import log2


class MRSemiSegDataset_tiger(SemiSegDataset):
    def __init__(self, datalist: str, pklpath: str, stage: str, data_ratio: dict, \
            root: str, patchsize: int, stridesize: int, tifpage: int, classes: int, \
            transform=None, preprocess=None, lrratio=8, islrmask=False, trainWSIratio: dict = ...):
        super().__init__(datalist, pklpath, stage, data_ratio, root, patchsize, stridesize, tifpage, classes, transform, preprocess, trainWSIratio)
        self.lrratio = lrratio
        self.lrshift = self.patchsize // 2 - self.patchsize // self.lrratio // 2
        self.lrdiff = int(log2(lrratio))
        self.islrmask = islrmask

        # print(self.lrratio, self.lrshift, self.lrdiff)

    def _tiffcheckcrop(self, slide, x, y, width, height):
        maxwidth, maxheight = slide.width, slide.height
        maxwidth -= x
        maxheight -= y
        if x>=0 and y>=0 and (width)<=maxwidth and (height)<=maxheight:
            image = self._tiffcrop(slide, x, y, width, height)
        else:
            w_comp = -x if x < 0 else 0
            h_comp = -y if y < 0 else 0
            w_crop = np.clip(width-w_comp, None, maxwidth)
            h_crop = np.clip(height-h_comp, None, maxheight)
            crop = self._tiffcrop(slide, np.clip(x, 0, None), np.clip(y, 0, None), w_crop, h_crop)

            image = np.full((height, width, 3), 255, dtype=np.uint8) if crop.shape[2] == 3 \
                else np.zeros((height, width, 1))
            image[h_comp:h_comp+h_crop, w_comp:w_comp+w_crop] = crop

        return image

    def __getitem__(self, idx):
        name, (x, y) = self.datalist[idx]

        if self.istrain:
            x += random.randint(0, self.stridesize)
            y += random.randint(0, self.stridesize)

        tifpath = os.path.join(self.root, 'images', f'{name}.tif')
        slide = pyvips.Image.tiffload(tifpath, page=self.tifpage)
        image = self._tiffcrop(slide, x, y, self.patchsize, self.patchsize)

        slide2 = pyvips.Image.tiffload(tifpath, page=self.tifpage+self.lrdiff)
        lrimage = self._tiffcheckcrop(
            slide2, x//self.lrratio-self.lrshift, y//self.lrratio-self.lrshift, self.patchsize, self.patchsize)


        if self.islabel:
            maskpath = os.path.join(self.root, 'tissue-masks', f'{name}.tif')
            slide = pyvips.Image.tiffload(maskpath, page=self.tifpage)        
            mask = self._tiffcrop(slide, x, y, self.patchsize, self.patchsize)

            if self.islrmask:
                slide2 = pyvips.Image.tiffload(maskpath, page=self.tifpage+self.lrdiff)
                lrmask = self._tiffcheckcrop(
                slide2, x//self.lrratio-self.lrshift, y//self.lrratio-self.lrshift, self.patchsize, self.patchsize)

                if self.transform:
                    sample = self.transform(image=image, mask=mask, lrimage=lrimage, lrmask=lrmask)
                    image, mask, lrimage, lrmask = sample['image'], sample['mask'], sample['lrimage'], sample['lrmask']

                if self.preprocess:
                    sample = self.preprocess(image=image, mask=mask, lrimage=lrimage, lrmask=lrmask)
                    image, mask, lrimage, lrmask = sample['image'], sample['mask'], sample['lrimage'], sample['lrmask']

                mask = torch.squeeze(mask, dim=0).long()
                lrmask = torch.squeeze(lrmask, dim=0).long()

                return image, mask, lrimage, lrmask

            if self.transform:
                sample = self.transform(image=image, mask=mask, lrimage=lrimage)
                image, mask, lrimage = sample['image'], sample['mask'], sample['lrimage']

            if self.preprocess:
                sample = self.preprocess(image=image, mask=mask, lrimage=lrimage)
                image, mask, lrimage = sample['image'], sample['mask'], sample['lrimage']

            mask = torch.squeeze(mask, dim=0).long()

            return image, mask, lrimage

        else:
            if self.transform:
                sample = self.transform(image=image, lrimage=lrimage)
                image, lrimage = sample['image'], sample['lrimage']

            if self.preprocess:
                sample = self.preprocess(image=image, lrimage=lrimage)
                image, lrimage = sample['image'], sample['lrimage']

            return image, lrimage
