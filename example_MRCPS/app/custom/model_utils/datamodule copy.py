import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from utils import SemiSegDataset, MRSemiSegDataset, get_strong_aug, get_weak_aug, get_preprocess
import random
class DataModule(pl.LightningDataModule):
    def __init__(self, traincfg: dict, client_id=0):
        super().__init__()
        self.traincfg = traincfg
        self.preprocess = get_preprocess()
        self.num_workers = 4
        self.client_id = client_id
        print("DataModule_ClientID", client_id)
        
        self.dataset =  MRSemiSegDataset

        #try to run
        # self.train_data_ratio = {
        #     'white_background': 0.0002,
        #     'tissue_background': 0.01,
        #     'whole_frontground': 0.01,
        #     'partial_frontground': 0.01,
        #     'partial_tissue': 0.01,
        #     'partial_tissue_wtarget': 0.01
        # }
        # self.test_data_ratio = {
        #     'white_background': 0,
        #     'tissue_background': 0.01,
        #     'whole_frontground': 0.01,
        #     'partial_frontground': 0.01,
        #     'partial_tissue': 0.01,
        #     'partial_tissue_wtarget': 0.01
        # }
        
        self.train_data_ratio = {
            'white_background': 0.0002,
            'tissue_background': 1,
            'whole_frontground': 1,
            'partial_frontground': 1,
            'partial_tissue': 1,
            'partial_tissue_wtarget': 1
        }
        self.test_data_ratio = {
            'white_background': 0,
            'tissue_background': 1,
            'whole_frontground': 1,
            'partial_frontground': 1,
            'partial_tissue': 1,
            'partial_tissue_wtarget': 1
        }
        self.label_batchsize = traincfg['traindl']['batchsize'] // 2 // 4 * 4
        self.unlabel_batchsize = traincfg['traindl']['batchsize'] - self.label_batchsize
        print(f"label batchsize: {self.label_batchsize}\tunlabel batchsize: {self.unlabel_batchsize}")

    def setup(self, stage=None):
        """
        Called at the beginning of fit (train + validate), validate, test, or predict.
        Args: 
            stage 'fit', 'validate', 'test', 'predict'
        """
        # Partition by client flag
        # for id in range(3):
        #     if self.client_id==id:



        # if self.client_id ==0 and "SemiSegPathology" in self.traincfg['rootset']['dataroot']: # KVGH
        #     train_ratio = {
        #             'KVGHlabel':[0,0], #1,
        #             'KVGHunlabel':[0,0], #self.traincfg['expset']['KVGHunlabel'],
        #             'NCKUlabel':[0,5], #0,
        #             'NCKUunlabel':[15,85]  #0
        #             } if "SemiSegPathology" in self.traincfg['rootset']['dataroot'] \
        #             else {
        #             'labelWSI':self.traincfg['expset']['labelWSI'],
        #             'totalWSI':self.traincfg['expset']['totalWSI'],
        #             }
            
        # elif self.client_id==1 and "SemiSegPathology" in self.traincfg['rootset']['dataroot']:
        #     train_ratio = {
        #             'KVGHlabel':[0,5], #0,
        #             'KVGHunlabel':[15,85], #0,
        #             'NCKUlabel':[0,0], #1,
        #             'NCKUunlabel':[0,0] #self.traincfg['expset']['NCKUunlabel']
        #             } if "SemiSegPathology" in self.traincfg['rootset']['dataroot'] \
        #             else {
        #             'labelWSI':self.traincfg['expset']['labelWSI'],
        #             'totalWSI':self.traincfg['expset']['totalWSI'],
        #             }
        # elif self.client_id==2 and "SemiSegPathology" in self.traincfg['rootset']['dataroot']:
        #     train_ratio = {
        #             'KVGHlabel':[5,7] ,#1,
        #             'KVGHunlabel':[85,113], #self.traincfg['expset']['KVGHunlabel'],
        #             'NCKUlabel':[5,8], #1,
        #             'NCKUunlabel':[85,127] #self.traincfg['expset']['NCKUunlabel']
        #             } if "SemiSegPathology" in self.traincfg['rootset']['dataroot'] \
        #             else {
        #             'labelWSI':self.traincfg['expset']['labelWSI'],
        #             'totalWSI':self.traincfg['expset']['totalWSI'],
        #             }
        
        
        settings = {
            "root":self.traincfg['rootset']['dataroot'],
            "preprocess":get_preprocess(),
            "datalist":'./dataset/fold_h.json',
            "classes":len(self.traincfg['classes']),
        }

        settings_train = {
            "pklpath":self.traincfg['rootset']['pklroot_train'],
            "patchsize":self.traincfg['traindl']['patchsize'],
            "stridesize":self.traincfg['traindl']['stridesize'],
            "tifpage":self.traincfg['traindl']['tifpage'],
        }
        if self.traincfg['traindl'].get('islrmask', False):
            settings_train['islrmask'] = True
        
        settings_test = {
            "pklpath":self.traincfg['rootset']['pklroot_test'],
            "patchsize":self.traincfg['testdl']['patchsize'],
            "stridesize":self.traincfg['testdl']['stridesize'],
            "tifpage":self.traincfg['testdl']['tifpage'],
        }
        if self.traincfg['testdl'].get('islrmask', False):
            settings_test['islrmask'] = True

        if 'mr' in self.traincfg['sslset']['type']:
            settings_train['lrratio'] = self.traincfg['expset'].get('lrratio', 8)
            settings_test['lrratio'] = self.traincfg['expset'].get('lrratio', 8)

        if stage == 'fit':
            label_aug = get_strong_aug() if self.traincfg['traindl'].get('sda', False) \
                    else get_weak_aug()

            unlabel_aug = get_weak_aug()
            self.train_label_dataset = self.dataset(
                stage='train_label',
                data_ratio=self.train_data_ratio,
                transform=label_aug,
                **settings_train,
                **settings
                )

            if 'sup' not in self.traincfg['sslset']['type'] and\
                '_f' not in self.traincfg['sslset']['type']:
                self.train_unlabel_dataset = self.dataset(
                    stage='train_unlabel',
                    data_ratio=self.train_data_ratio,
                    transform=unlabel_aug,
                    **settings_train,
                    **settings
                    )

            self.valid_dataset = self.dataset(
                stage='valid',
                data_ratio=self.test_data_ratio,
                **settings_test,
                **settings
                )
            ## add test data why not?
            self.test_dataset = self.dataset(
                stage='test',
                data_ratio=self.test_data_ratio,
                **settings_test,
                **settings
                )
            self.train_size = len(self.train_label_dataset)+len(self.train_unlabel_dataset)
        if stage == 'test':
            self.test_dataset = self.dataset(
                stage='test',
                data_ratio=self.test_data_ratio,
                **settings_test,
                **settings
                )

    def train_dataloader(self):
        settings = {
            "shuffle":True,
            "drop_last":True,
            "pin_memory":True,
            "persistent_workers":True,
        }
        ## get dataset subset
        num_of_samples_l = len(self.train_label_dataset)//4
        subset_l = list(range(0, len(self.train_label_dataset)))
        random.shuffle(subset_l)
        subset_l = subset_l[0:num_of_samples_l]
        
        num_of_samples_u = len(self.train_unlabel_dataset)//4
        subset_u = list(range(0, len(self.train_unlabel_dataset)))
        random.shuffle(subset_u)
        subset_u = subset_u[0:num_of_samples_u]
        
        trainset_l = torch.utils.data.Subset(self.train_label_dataset, subset_l)
        trainset_u = torch.utils.data.Subset(self.train_unlabel_dataset, subset_u)
        
        
        if 'sup' in self.traincfg['sslset']['type'] or\
            '_f' in self.traincfg['sslset']['type']:
            labeled_dataloader = DataLoader(
                # dataset=self.train_label_dataset,
                dataset=trainset_l, # use subset
                batch_size = self.traincfg['traindl']['batchsize'],
                num_workers=self.num_workers,
                **settings
                )
            return {"label": labeled_dataloader}

        else:
            labeled_dataloader = DataLoader(
                # dataset=self.train_label_dataset,
                dataset=trainset_l,  # use subset
                batch_size = self.label_batchsize,
                num_workers = self.num_workers,
                **settings
                )
            unlabeled_dataloader = DataLoader(
                # dataset=self.train_unlabel_dataset,
                dataset=trainset_u,  # use subset
                batch_size = self.unlabel_batchsize,
                num_workers = self.num_workers,
                **settings
                )
            
        return {"label": labeled_dataloader, "unlabel": unlabeled_dataloader}

    def val_dataloader(self):
        
        # num_of_samples_l = len(self.train_label_dataset)/4
        # subset_l = list(range(0, len(self.train_label_dataset)))
        # subset_l = random.shuffle(subset_l)
        # subset_l = subset_l[0:num_of_samples_l]
        
        # subset = list(range(0, len(self.valid_dataset)//4))
        # sub_val = torch.utils.data.Subset(self.valid_dataset, subset)
        
        return DataLoader(
                dataset=self.valid_dataset,
                # dataset = sub_val,
                batch_size = self.traincfg['testdl']['batchsize'],
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
                )

    def test_dataloader(self):
        
        subset = list(range(0, len(self.test_dataset)//4))
        # subset = list(range(0, 100))
        sub_test = torch.utils.data.Subset(self.test_dataset, subset)
        
        return DataLoader(
                dataset=self.test_dataset,
                # dataset=sub_test,
                batch_size= self.traincfg['testdl']['batchsize'],
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
                )
