import os
import pyvips
import numpy as np
import torch
import torch.nn.functional as F
import random
from used_models.MRCPS.utils.dataset import SemiSegDataset
from math import log2


'''
mutiscale crop

'''

class MRSemiSegDataset(SemiSegDataset):
    def __init__(self, datalist: str, pklpath: str, stage: str, data_ratio: dict, \
            root: str, tifroot:str, maskroot:str, patchsize: int, stridesize: int, tifpage: int, \
            classes: int, transform=None, preprocess=None, lrratio=8, islrmask=False):
        super().__init__(datalist, pklpath, stage, data_ratio, root, tifroot, maskroot, \
                          patchsize, stridesize, tifpage, classes, transform, preprocess)
        self.lrratio = lrratio
        #point in low scale need / ratio, but not real location of patch as lefttop 
        self.lrshift = self.patchsize // 2 - self.patchsize // self.lrratio // 2    #distance between patch and patch/lrratio
        self.lrdiff = int(log2(lrratio))
        self.islrmask = islrmask
        self.stage = stage
        # print(self.lrratio, self.lrshift, self.lrdiff)

    def _tiffcheckcrop(self, slide, x, y, width, height):
        maxwidth, maxheight = slide.width, slide.height
        maxwidth -= x
        maxheight -= y
        # print(f'>>{slide.width},{slide.height}, {x}, {y}, {width}, {height}')
        if x>=0 and y>=0 and (width)<=maxwidth and (height)<=maxheight:
            image = self._tiffcrop(slide, x, y, width, height)
        else:
            # lefttop smaller than 0 crop location start from 0,0 but not over largest size
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
        #get filename and location
        name, (x, y) = self.datalist[idx]

        # tifpath = os.path.join(self.root, 'tifs', f'{name}.tif')
        tifpath = os.path.join(self.tifroot, f'{name}.tif')
        # tifpath = os.path.join(self.tifroot, f'{name}')
        slide = pyvips.Image.tiffload(tifpath, page=self.tifpage)

        #add random stride to slice batch initial point
        if self.istrain:
            x += random.randint(0, self.stridesize)
            y += random.randint(0, self.stridesize)
            #prevent out boundary
            if (x+self.patchsize)>slide.width:
                x=slide.width-self.patchsize
            if (y+self.patchsize)>slide.height:
                y=slide.height-self.patchsize

        #crop patch
        image = self._tiffcrop(slide, x, y, self.patchsize, self.patchsize)

        #crop large scale patch
        slide2 = pyvips.Image.tiffload(tifpath, page=self.tifpage+self.lrdiff)
        lrimage = self._tiffcheckcrop(  
            slide2, x//self.lrratio-self.lrshift, y//self.lrratio-self.lrshift, self.patchsize, self.patchsize)


        if self.islabel:
            # if use tif mask
            # maskpath = os.path.join(self.root, 'masks', f'{name}.tif')
            maskpath = os.path.join(self.maskroot, f'{name}.tif')
            # maskpath = os.path.join(self.maskroot, f'{name}')
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
