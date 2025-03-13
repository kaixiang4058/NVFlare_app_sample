import os
from math import log2
import random

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pyvips
import pickle


'''
mutiscale crop

'''

class HisPathDataset(Dataset):
    def __init__(self, datalist: str, pklpath: str, stage: str, data_ratio: dict, \
            root: str, tifroot:str, maskroot:str, patchsize: int, stridesize: int, tifpage: int, \
            classes: list, transform=None, preprocess=None, lrratio=8, islrmask=False):
        super().__init__()

        #--load chosen file json--
        totaldict = self._loading_data(datalist, pklpath, stage)    #load data name location from pkl
        
        #--select data with ratio --
        self.datalist = self._select_data(totaldict, data_ratio)    #select data with ratio

        self.root = root
        self.tifroot = tifroot
        self.maskroot = maskroot

        self.patchsize = patchsize
        self.stridesize = stridesize
        self.tifpage = tifpage
        self.classes = classes

        self.transform = transform
        self.preprocess = preprocess

        self.istrain = True if "train" in stage else False
        self.islabel = False if stage == "train_unlabel" else True


        self.lrratio = lrratio
        #point in low scale need / ratio, but not real location of patch as lefttop 
        self.lrshift = self.patchsize // 2 - self.patchsize // self.lrratio // 2    #distance between patch and patch/lrratio
        self.lrdiff = int(log2(lrratio))
        self.islrmask = islrmask
        self.stage = stage
        # print(self.lrratio, self.lrshift, self.lrdiff)

        
    def __len__(self):
        return len(self.datalist)
    
   #-- read selection of file from json (with stage, label/unlabel) --
    def readDataList(self, stage:str, 
                    datalist = "/home/u7085556/SemiSegPathology/dataset/fold_h.json"):
        with open(datalist) as fp:
            dataset = json.load(fp)
        
        case = []

        if stage == "train_label" and 'label' in dataset['train']:
            case += dataset['train']['label']
        elif stage == "train_unlabel" and 'unlabel' in dataset['train']:
            case += dataset['train']['unlabel']
        elif stage == "valid" or stage=='test':
            case += dataset[stage]['label']
        # else:
        #     case += dataset[stage]['label']
            
        print(f"{stage} with {len(case)} cases.")
        # print(case)

        return case

    # -- load data information from pkl --
    def _loading_data(self, datalist, pklpath, stage):
        #get file list
        cases = self.readDataList(stage, datalist)

        #load data from pkl
        totaldict = {
            'white_background' : [],
            'tissue_background' : [],
            'whole_frontground' : [],
            'partial_frontground' : [],
            'partial_tissue' : [],
            'partial_tissue_wtarget' : []
        }
        for case in cases:
            with open(os.path.join(pklpath, f'{case}.pkl'), 'rb') as f:
                pkldict = pickle.load(f)
            for key in totaldict:
                totaldict[key] += pkldict[key]

        #show number of pkl info
        print('---data amount of different type patches ---')
        for key in totaldict:
            print(key, ':', len(totaldict[key]))

        return totaldict

 
    def _select_data(self, pkldict, ratio):
        datalist = []
        label_keys = ['whole_frontground', 'partial_frontground', 'partial_tissue_wtarget']
        unlabel_keys = ['white_background', 'tissue_background', 'partial_tissue']

#         # limit = {
#         #     'white_background': [10, 1000],
#         #     'tissue_background': [1000, 100000],
#         #     'whole_frontground': [0, 150000],                #label
#         #     'partial_frontground': [0, 150000],              #label
#         #     'partial_tissue': [1000, 100000],
#         #     'partial_tissue_wtarget': [0, 0]     #label
#         # }
#         limit = {
#             'white_background': [10, 10],
#             'tissue_background': [1000, 20000],
#             'whole_frontground': [0, 20000],                #label
#             'partial_frontground': [0, 20000],              #label
#             'partial_tissue': [1000, 20000],
#             'partial_tissue_wtarget': [0, 0]     #label
#         }
#         # limit = {
#         #     'white_background': [10, 100],
#         #     'tissue_background': [10, 100],
#         #     'whole_frontground': [0, 100],                #label
#         #     'partial_frontground': [0, 100],              #label
#         #     'partial_tissue': [10, 100],
#         #     'partial_tissue_wtarget': [0, 0]     #label
#         # }
        

#         if ratio['whole_frontground']!=0: #label set 
#             foreground_datas=[]   
#             for k in label_keys:
#                 sample_num = int(len(pkldict[k])*ratio[k])
#                 if sample_num <= 0:
#                     # print('label:',k , 'no data')
#                     sample_num = 0
#                 if sample_num > limit[k][1]:    #limit number lower 100000
#                     sample_num = limit[k][1]
#                 sample_datas = random.sample(pkldict[k], sample_num)    
#                 print(f'real input data {k} : {len(sample_datas)}')
#                 foreground_datas+=sample_datas
#             background_limit = len(foreground_datas)//2

#             background_datas=[]
#             for k in unlabel_keys:
#                 sample_num = int(len(pkldict[k])*ratio[k])
#                 # if sample_num>background_limit:
#                 #     sample_num=background_limit
#                 if sample_num > limit[k][1]:    #limit number lower 100000
#                     sample_num = limit[k][1]
#                 elif sample_num <= 0:       #avoid none data siguation
#                     sample_num = 0

#                 sample_datas = random.sample(pkldict[k], sample_num)
#                 print(f'real input data {k} : {len(sample_datas)}')
#                 background_datas+=sample_datas

#             datalist = foreground_datas+background_datas

#         elif ratio['whole_frontground']==0: #unlabel set
#             for k, v in ratio.items():
#                 sample_datas=[]
#                 if len(pkldict[k]) * v > limit[k][1]:    #limit number lower 100000
#                     sample_datas = random.sample(pkldict[k], limit[k][1])
#                 elif len(pkldict[k]) * v <= 0:       #avoid none data siguation
#                     sample_datas = []
#                 elif len(pkldict[k]) * v < limit[k][0] :   #keep least self if smaller 1000 
#                     if len(pkldict[k]) >limit[k][0]:
#                         sample_datas = random.sample(pkldict[k], limit[k][0])
#                     else:
#                         sample_datas = pkldict[k]
#                 else:
#                     sample_datas = random.sample(pkldict[k], int(len(pkldict[k]) * v))
#                 print(f'real input data {k} : {len(sample_datas)}')
#                 datalist+=sample_datas


#         for _ in range(10):
#             np.random.shuffle(datalist)
#         return datalist

        # limit = {
        #     'white_background': [10, 10],
        #     'tissue_background': [10, -1],
        #     'whole_frontground': [0, -1],                #label
        #     'partial_frontground': [0, -1],              #label
        #     'partial_tissue': [10, -1],
        #     'partial_tissue_wtarget': [0, -1]     #label
        # }

        # limit = {
        #     'white_background': [10, 1000],
        #     'tissue_background': [1000, 150000],
        #     'whole_frontground': [0, 150000],                #label
        #     'partial_frontground': [0, 150000],              #label
        #     'partial_tissue': [1000, 100000],
        #     'partial_tissue_wtarget': [0, 0]     #label
        # }

        # limit = {
        #     'white_background': [10, 1000],
        #     'tissue_background': [1000, 100000],
        #     'whole_frontground': [0, 100000],                #label
        #     'partial_frontground': [0, 100000],              #label
        #     'partial_tissue': [1000, 50000],
        #     'partial_tissue_wtarget': [0, 0]     #label
        # }

        limit = {
            'white_background': [10, 1000],
            'tissue_background': [1000, 10000],
            'whole_frontground': [0, 10000],                #label
            'partial_frontground': [0, 10000],              #label
            'partial_tissue': [1000, 10000],
            'partial_tissue_wtarget': [0, 0]     #label
        }
        
        # for test
        # limit = {
        #     'white_background': [10, 100],
        #     'tissue_background': [10, 100],
        #     'whole_frontground': [0, 100],                #label
        #     'partial_frontground': [0, 100],              #label
        #     'partial_tissue': [10, 100],
        #     'partial_tissue_wtarget': [0, 100]     #label
        # }

        if ratio['whole_frontground']!=0: #label set 
            foreground_datas=[]   
            for k in label_keys:
                sample_num = int(len(pkldict[k])*ratio[k])
                if sample_num <= 0:
                    # print('label:',k , 'no data')
                    sample_num = 0
                # if sample_num > limit[k][1]:    #limit number lower 100000
                #     sample_num = limit[k][1]
                sample_datas = random.sample(pkldict[k], sample_num)    
                print(f'real input data {k} : {len(sample_datas)}')
                foreground_datas+=sample_datas
            background_limit = len(foreground_datas)//4

            background_datas=[]
            for k in unlabel_keys:
                sample_num = int(len(pkldict[k])*ratio[k])
                #limit backgorund tissue large than foreground
                if sample_num>background_limit:
                    sample_num=background_limit
                
                if limit[k][1] == -1:
                    sample_num = sample_num
                elif sample_num > limit[k][1]:    #limit number lower 100000
                    sample_num = limit[k][1]
                elif sample_num <= 0:       #avoid none data siguation
                    sample_num = 0

                sample_datas = random.sample(pkldict[k], sample_num)
                print(f'real input data {k} : {len(sample_datas)}')
                background_datas+=sample_datas

            datalist = foreground_datas+background_datas

        elif ratio['whole_frontground']==0: #unlabel set
            for k, v in ratio.items():
                sample_datas=[]
                
                if limit[k][1] == -1:
                    sample_datas = pkldict[k]
                elif len(pkldict[k]) * v > limit[k][1]:    #limit number lower 100000
                    sample_datas = random.sample(pkldict[k], limit[k][1])
                elif len(pkldict[k]) * v <= 0:       #avoid none data siguation
                    sample_datas = []
                elif len(pkldict[k]) * v < limit[k][0] :   #keep least self if smaller 1000 
                    if len(pkldict[k]) >limit[k][0]:
                        sample_datas = random.sample(pkldict[k], limit[k][0])
                    else:
                        sample_datas = pkldict[k]
                else:
                    sample_datas = random.sample(pkldict[k], int(len(pkldict[k]) * v))
                print(f'real input data {k} : {len(sample_datas)}')
                datalist+=sample_datas


        for _ in range(10):
            np.random.shuffle(datalist)
        return datalist


    def read_WSI(self, wsi_path, data_type, level):
        slice = None
        if data_type == 'tif':
            slice = pyvips.Image.tiffload(wsi_path, page=level)
        elif data_type in ['svs','ndpi','mrxs']:
            slice = pyvips.Image.new_from_file(wsi_path, level=level)
        else:
            print('type not include')
        return slice




    def _tiffcrop(self, slide, x, y, width, height):
        #crop
        # vi = slide.crop(x, y, width, height)    #lefttop
        # image = np.ndarray(buffer=vi.write_to_memory(),
        #                  dtype=np.uint8,
        #                  shape=[vi.height, vi.width, vi.bands])

        #fetch (more fast)
        region = pyvips.Region.new(slide)
        vi = region.fetch(x, y, width, height)
        image = np.ndarray(buffer=vi,
                            dtype=np.uint8,
                            shape=[height, width, slide.bands])
        
        if image.shape[2] == 4:
            image = image[:, :, 0:3]
        return image
    
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
        # name, (x, y) , datatype= self.datalist[idx] ＃old don't have datatype
        name, (x, y) = self.datalist[idx]
        datatype = 'tif'
        # tifpath = os.path.join(self.root, 'tifs', f'{name}.tif')
        tifpath = os.path.join(self.tifroot, f'{name}.{datatype}')
        # tifpath = os.path.join(self.tifroot, f'{name}')
        slide = self.read_WSI(tifpath, datatype, level=self.tifpage)
        # slide = pyvips.Image.tiffload(tifpath, page=self.tifpage)

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
        slide2 = self.read_WSI(tifpath, datatype, level=self.tifpage)
        # slide2 = pyvips.Image.tiffload(tifpath, page=self.tifpage+self.lrdiff)
        lrimage = self._tiffcheckcrop(  
            slide2, x//self.lrratio-self.lrshift, y//self.lrratio-self.lrshift, self.patchsize, self.patchsize)

        if self.islabel:
            # if use tif mask
            # maskpath = os.path.join(self.root, 'masks', f'{name}.tif')
            maskpath = os.path.join(self.maskroot, f'{name}.tif')
            # maskpath = os.path.join(self.maskroot, f'{name}')
            slide_mask = self.read_WSI(maskpath, 'tif', level=self.tifpage)
            # slide_mask = pyvips.Image.tiffload(maskpath, page=self.tifpage)
            mask = self._tiffcrop(slide_mask, x, y, self.patchsize, self.patchsize)

            if self.islrmask:
                slide2_mask = self.read_WSI(maskpath, 'tif', level=self.tifpage+self.lrdiff)
                # slide2 = pyvips.Image.tiffload(maskpath, page=self.tifpage+self.lrdiff)
                lrmask = self._tiffcheckcrop(
                slide2_mask, x//self.lrratio-self.lrshift, y//self.lrratio-self.lrshift, self.patchsize, self.patchsize)

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



