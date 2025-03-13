import os
import json
import pyvips
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import random

# +
# [tiff_name, (x, y)] -> train_list
# -

'''
1.load file with pkl
    -check label/unlabel
    -assign with patch type
2.select data with ratio
3.set parameter(for inherit)

with origin __getitem__ and _tiffcrop function
'''

class SemiSegDataset(Dataset):
    def __init__(self, datalist:str, pklpath:str, stage:str, data_ratio:dict, root:str, tifroot, maskroot,
                 patchsize:int, stridesize:int, tifpage:int, classes:int, transform=None, preprocess=None
                 ):
        super().__init__()
        print('*****load data pkl')
        totaldict = self._loading_data(datalist, pklpath, stage)    #load data name location from pkl
        print('*****select from pkl')
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

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        name, (x, y) = self.datalist[idx]

        if self.istrain:
            x += random.randint(0, self.stridesize)
            y += random.randint(0, self.stridesize)
        tifpath = os.path.join(self.root, 'tifs', f'{name}.tif')
        # https://github.com/libvips/pyvips/blob/2bfee2aea934c875424ed97316470b89bb86935a/doc/vimage.rst
        slide = pyvips.Image.tiffload(tifpath, page=self.tifpage)
        image = self._tiffcrop(slide, x, y, self.patchsize, self.patchsize)

        if self.islabel:
            maskpath = os.path.join(self.root, 'masks', f'{name}.tif')
            slide = pyvips.Image.tiffload(maskpath, page=self.tifpage)
            mask = self._tiffcrop(slide, x, y, self.patchsize, self.patchsize)

            if self.transform:
                sample = self.transform(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            if self.preprocess:
                sample = self.preprocess(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            return image, torch.squeeze(mask, dim=0).long()
        else:
            if self.transform:
                sample = self.transform(image=image)
                image = sample['image']

            if self.preprocess:
                sample = self.preprocess(image=image)
                image = sample['image']

            return image

        # image = torch.zeros([3, 5, 5])
        # mask = torch.zeros([1, 5, 5])
        # return image, mask

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

    def _loading_data(self, datalist, pklpath, stage):
        #read file name from json
        cases = readDataList(stage, datalist)

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

        # print('---data amount of different type patches ---')
        for key in totaldict:
            print(key, ':', len(totaldict[key]))

        return totaldict

    def _select_data(self, pkldict, ratio):
        datalist = []
        label_keys = ['whole_frontground', 'partial_frontground', 'partial_tissue_wtarget']
        unlabel_keys = ['white_background', 'tissue_background', 'partial_tissue']

        # limit = {
        #     'white_background': [10, 1000],
        #     'tissue_background': [1000, 150000],
        #     'whole_frontground': [0, 150000],                #label
        #     'partial_frontground': [0, 150000],              #label
        #     'partial_tissue': [1000, 100000],
        #     'partial_tissue_wtarget': [0, 0]     #label
        # }

        limit = {
            'white_background': [10, 1000],
            'tissue_background': [1000, 100000],
            'whole_frontground': [0, 100000],                #label
            'partial_frontground': [0, 100000],              #label
            'partial_tissue': [1000, 50000],
            'partial_tissue_wtarget': [0, 0]     #label
        }

        # limit = {
        #     'white_background': [10, 1000],
        #     'tissue_background': [1000, 50000],
        #     'whole_frontground': [0, 100000],                #label
        #     'partial_frontground': [0, 100000],              #label
        #     'partial_tissue': [1000, 25000],
        #     'partial_tissue_wtarget': [0, 0]     #label
        # }
        
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
            background_limit = len(foreground_datas)//2

            background_datas=[]
            for k in unlabel_keys:
                sample_num = int(len(pkldict[k])*ratio[k])
                if sample_num>background_limit:
                    sample_num=background_limit
                if sample_num > limit[k][1]:    #limit number lower 100000
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
                if len(pkldict[k]) * v > limit[k][1]:    #limit number lower 100000
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

#read file from json
def readDataList(stage:str, 
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

