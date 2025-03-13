import os
from math import log2
import random

import numpy as np
import yaml
import json
import pyvips
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# from used_models.MRCPS.utils import get_strong_aug, get_weak_aug, get_preprocess
from dataset.transform import get_strong_aug, get_weak_aug, get_preprocess
# from dataset.HisPathDataset import HisPathDataset
from dataset.HisPathDataset_hack import HisPathDataset

import pytorch_lightning as pl

class DataModule(pl.LightningDataModule):
    def __init__(self, cfgpath: str, client_id=0):
        '''
        Perpare train, valid, test dataloader
        '''
        super().__init__()

        #--Load config--
        with open(cfgpath, 'r') as fp:
            self.traincfg = yaml.load(fp, Loader=yaml.FullLoader)
        
        #--For federated learning--
        # self.client_id = client_id
        # print("DataModule_ClientID", client_id)

        self.preprocess = get_preprocess()
        self.dataset =  HisPathDataset
        self.num_workers = 4

        # self.label_batchsize = self.traincfg['traindl']['batchsize'] // 2 // 4 * 4
        # self.unlabel_batchsize = self.traincfg['traindl']['batchsize'] - self.label_batchsize
        self.label_batchsize = self.traincfg['traindl']['batchsize']
        self.unlabel_batchsize = self.traincfg['traindl']['batchsize']
        print(f"label batchsize: {self.label_batchsize}\tunlabel batchsize: {self.unlabel_batchsize}")


        # ratio of input
        self.train_data_ratio = {
            'white_background': 0.0001,
            'tissue_background': 1,
            'whole_frontground': 1,
            'partial_frontground': 1,
            'partial_tissue': 1,
            'partial_tissue_wtarget': 1
        }

        self.train_unlabel_data_ratio = {
            'white_background': 0,
            'tissue_background': 1,
            'whole_frontground': 0,         #no front
            'partial_frontground': 0,       #no front
            'partial_tissue': 1,
            'partial_tissue_wtarget': 0     #no target
        }

        self.test_data_ratio = {
            'white_background': 0,
            'tissue_background': 1,
            'whole_frontground': 1,
            'partial_frontground': 1,
            'partial_tissue': 1,
            'partial_tissue_wtarget': 1
        }

    def setup(self, stage=None):
        settings = {
            "root":self.traincfg['rootset']['dataroot'],
            "tifroot":self.traincfg['rootset']['tifroot'],
            "maskroot":self.traincfg['rootset']['maskroot'],
            "datalist":self.traincfg['rootset']['datalist'],
            "classes":len(self.traincfg['classes']),
            "patchsize":self.traincfg['traindl']['patchsize'],
            "stridesize":self.traincfg['traindl']['stridesize'],
            "tifpage":self.traincfg['traindl']['tifpage'],
            "preprocess":get_preprocess()
        }

        if stage == 'fit':
            print('--------------')
            label_aug = get_strong_aug() if self.traincfg['traindl'].get('sda', False) \
                    else get_weak_aug()

            unlabel_aug = get_weak_aug()

            # print('*****training label dataset')
            self.train_label_dataset = self.dataset(
                stage='train_label',
                pklpath = self.traincfg['rootset']['pklroot_label'],
                data_ratio=self.train_data_ratio,
                transform=label_aug,
                **settings
                )
            print(self.train_label_dataset)

            # print('*****training unlabel dataset')
            self.train_unlabel_dataset = self.dataset(
                stage='train_unlabel',
                pklpath = self.traincfg['rootset']['pklroot_unlabel'],
                data_ratio=self.train_unlabel_data_ratio,
                transform=unlabel_aug,
                **settings
                )
            self.train_size = len(self.train_label_dataset)+len(self.train_unlabel_dataset)
            
        # print('*****training valid dataset')
        self.valid_dataset = self.dataset(
            stage='valid',
            pklpath = self.traincfg['rootset']['pklroot_label'],
            data_ratio=self.test_data_ratio,
            **settings
            )
        
        # print('*****training test dataset')
        self.test_dataset = self.dataset(
            stage='test',
            pklpath = self.traincfg['rootset']['pklroot_label'],
            data_ratio=self.test_data_ratio,
            **settings
            )
        

    def train_dataloader(self):
        # print('*****training dataloader')
        settings = {
            "shuffle":True,
            "drop_last":True,
            "pin_memory":True,
            "persistent_workers":True,
        }
        ## get dataset subset
        num_of_samples_l = len(self.train_label_dataset)
        # subset_l = list(range(0, len(self.train_label_dataset)))
        # random.shuffle(subset_l)
        # subset_l = subset_l[0:num_of_samples_l]
        # trainset_l = torch.utils.data.Subset(self.train_label_dataset, subset_l)
        
        num_of_samples_u = len(self.train_unlabel_dataset)
        # subset_u = list(range(0, len(self.train_unlabel_dataset)))
        # random.shuffle(subset_u)
        # subset_u = subset_u[0:num_of_samples_u]
        # trainset_u = torch.utils.data.Subset(self.train_unlabel_dataset, subset_u)
        
        dataloader_dict = {}

        if num_of_samples_l>0:
            labeled_dataloader = DataLoader(
                dataset=self.train_label_dataset,
                # dataset=trainset_l,  # use subset
                batch_size = self.label_batchsize,
                num_workers = self.num_workers,
                **settings
                )
            dataloader_dict['label'] = labeled_dataloader
        if num_of_samples_u>0:
            unlabeled_dataloader = DataLoader(
                dataset=self.train_unlabel_dataset,
                # dataset=trainset_u,  # use subset
                batch_size = self.unlabel_batchsize,
                num_workers = self.num_workers,
                **settings
                )
            dataloader_dict['unlabel'] = unlabeled_dataloader
        return dataloader_dict

    def val_dataloader(self):
        # print('*****valid dataloader')
        return DataLoader(
                dataset=self.valid_dataset,
                # dataset = sub_val,
                batch_size = self.traincfg['testdl']['batchsize'],
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
                )

    def test_dataloader(self):
        # print('*****test dataloader')
        # subset = list(range(0, 100))
        # sub_test = torch.utils.data.Subset(self.test_dataset, subset)
        
        return DataLoader(
                dataset=self.test_dataset,
                # dataset=sub_test,
                batch_size= self.traincfg['testdl']['batchsize'],
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
                )



