import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from used_models.MRCPS.utils import SemiSegDataset, MRSemiSegDataset, get_strong_aug, get_weak_aug, get_preprocess
import random
class DataModule(pl.LightningDataModule):
    def __init__(self, traincfg: dict, client_id=0):
        super().__init__()
        self.traincfg = traincfg #datacfg sslset, traindl, testdl, rootset, expset
        self.preprocess = get_preprocess()
        self.num_workers = 4
        self.client_id = client_id
        print("DataModule_ClientID", client_id)
        
        # choose dataset code
        self.dataset =  MRSemiSegDataset if 'mr' in traincfg['sslset']['type'] or 'ms' in traincfg['sslset']['type'] \
            else SemiSegDataset
            
        # ratio of input
        # self.train_data_ratio = {
        #     'white_background': 0,
        #     'tissue_background': 0.01,
        #     'whole_frontground': 0.01,
        #     'partial_frontground': 0.01,
        #     'partial_tissue': 0.01,
        #     'partial_tissue_wtarget': 0.01
        # }
        # self.train_unlabel_data_ratio = {
        #     'white_background': 0,
        #     'tissue_background': 0.005,
        #     'whole_frontground': 0,
        #     'partial_frontground': 0,
        #     'partial_tissue': 0.005,
        #     'partial_tissue_wtarget': 0.005
        # }
        # self.test_data_ratio = {
        #     'white_background': 0,
        #     'tissue_background': 0.02,
        #     'whole_frontground': 0.02,
        #     'partial_frontground': 0.02,
        #     'partial_tissue': 0.02,
        #     'partial_tissue_wtarget': 0.02
        # }
        
        self.train_data_ratio = {
            'white_background': 0.0001,
            'tissue_background': 1,
            'whole_frontground': 1,
            'partial_frontground': 1,
            'partial_tissue': 1,
            'partial_tissue_wtarget': 1
        }

        # self.train_unlabel_data_ratio = {
        #     'white_background': 0,
        #     'tissue_background': 0.5,
        #     'whole_frontground': 0,
        #     'partial_frontground': 0,
        #     'partial_tissue': 0.1,
        #     'partial_tissue_wtarget': 0
        # }
        self.train_unlabel_data_ratio = {
            'white_background': 0,
            'tissue_background': 1,
            'whole_frontground': 0,
            'partial_frontground': 0,
            'partial_tissue': 1,
            'partial_tissue_wtarget': 0
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
        # self.label_batchsize = 2
        # self.unlabel_batchsize = 2
        print(f"label batchsize: {self.label_batchsize}\tunlabel batchsize: {self.unlabel_batchsize}")

    def setup(self, stage=None):
        """
        Called at the beginning of fit (train + validate), validate, test, or predict.
        Args: 
            stage 'fit', 'validate', 'test', 'predict'
        """
        # Partition by client flag
        print(self.traincfg['rootset']['datalist'])
        settings = {
            "root":self.traincfg['rootset']['dataroot'],
            "tifroot":self.traincfg['rootset']['tifroot'],
            "maskroot":self.traincfg['rootset']['maskroot'],
            "preprocess":get_preprocess(),
            "datalist":self.traincfg['rootset']['datalist'],
            "classes":len(self.traincfg['classes']),
        }

        settings_train = {
            "pklpath":self.traincfg['rootset']['pklroot_train'],
            "patchsize":self.traincfg['traindl']['patchsize'],
            "stridesize":self.traincfg['traindl']['stridesize'],
            "tifpage":self.traincfg['traindl']['tifpage'],
        }
        # if self.traincfg['traindl'].get('islrmask', False):
        settings_train['islrmask'] = True
        

        settings_train_unlabel = {
            "pklpath":self.traincfg['rootset']['pklroot_train_unlabel'],
            "patchsize":self.traincfg['traindl']['patchsize'],
            "stridesize":self.traincfg['traindl']['stridesize'],
            "tifpage":self.traincfg['traindl']['tifpage'],
        }

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

            # if not 'unlabel' in self.traincfg['rootset']['datalist']:
            # print('*****training label dataset')
            self.train_label_dataset = self.dataset(
                stage='train_label',
                data_ratio=self.train_data_ratio,
                transform=label_aug,
                **settings_train,
                **settings
                )

            # print('*****training unlabel dataset')
            # if 'sup' not in self.traincfg['sslset']['type'] and\
            #     '_f' not in self.traincfg['sslset']['type']:
            self.train_unlabel_dataset = self.dataset(
                stage='train_unlabel',
                data_ratio=self.train_unlabel_data_ratio,
                transform=unlabel_aug,
                **settings_train_unlabel,
                **settings
                )

            
            # print('*****training valid dataset')
            self.valid_dataset = self.dataset(
                stage='valid',
                data_ratio=self.test_data_ratio,
                **settings_test,
                **settings
                )
            
            # print('*****training test dataset')
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
        
        # print('*****training dataloader')
        settings = {
            "shuffle":True,
            "drop_last":True,
            "pin_memory":True,
            "persistent_workers":True,
        }
        ## get dataset subset
        num_of_samples_l = len(self.train_label_dataset)
        subset_l = list(range(0, len(self.train_label_dataset)))
        random.shuffle(subset_l)
        subset_l = subset_l[0:num_of_samples_l]
        
        num_of_samples_u = len(self.train_unlabel_dataset)//4
        subset_u = list(range(0, len(self.train_unlabel_dataset)))
        random.shuffle(subset_u)
        subset_u = subset_u[0:num_of_samples_u]
        
        print('train loader sample number')
        print(num_of_samples_l)
        print(num_of_samples_u)
        trainset_l = torch.utils.data.Subset(self.train_label_dataset, subset_l)
        trainset_u = torch.utils.data.Subset(self.train_unlabel_dataset, subset_u)
        
        
        # if 'sup' in self.traincfg['sslset']['type'] or\
        #     '_f' in self.traincfg['sslset']['type']:
        #     labeled_dataloader = DataLoader(
        #         # dataset=self.train_label_dataset,
        #         dataset=trainset_l, # use subset
        #         batch_size = self.traincfg['traindl']['batchsize'],
        #         num_workers=self.num_workers,
        #         **settings
        #         )
        #     return {"label": labeled_dataloader}
        dataloader_dict = {}

        if num_of_samples_l>0:
            labeled_dataloader = DataLoader(
                # dataset=self.train_label_dataset,
                dataset=trainset_l,  # use subset
                batch_size = self.label_batchsize,
                num_workers = self.num_workers,
                **settings
                )
            dataloader_dict['label'] = labeled_dataloader
        if num_of_samples_u>0:
            # labeled_dataloader = DataLoader(
            #     # dataset=self.train_label_dataset,
            #     dataset=trainset_l,  # use subset
            #     batch_size = self.label_batchsize,
            #     num_workers = self.num_workers,
            #     **settings
            #     )
            unlabeled_dataloader = DataLoader(
                # dataset=self.train_unlabel_dataset,
                dataset=trainset_u,  # use subset
                batch_size = self.unlabel_batchsize,
                num_workers = self.num_workers,
                **settings
                )
            dataloader_dict['unlabel'] = unlabeled_dataloader
            
        return dataloader_dict

    def val_dataloader(self):
        
        # print('*****valid dataloader')
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
        
        # print('*****test dataloader')
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
