import torch
import pytorch_lightning as pl
from torch import nn
import numpy as np

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smputils
from transformers import get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

import copy

from model_utils import transformsgpu, transformmasks
import networks
        
import yaml
import os

class BaseModel(pl.LightningModule):
    def __init__(self, cfg_path, cfg_root=''):
        """
        traincfg structure:
        {
            rootset: {...}
            sslset: {...}
            expset: {...}
            loss: {...}
            branch{idx} : {...}
        }
        """
        super().__init__()
        print(os.getcwd())
        traincfg = self.load_modelcfg(cfg_path, cfg_root)

        self.save_hyperparameters(traincfg)
        self.traincfg = copy.deepcopy(traincfg)

        self.modelnum = self.traincfg['modelnum']

        # Unify the criterion method
        self.criterion = self._initloss()
        self.infidx = 0
        # self.metrics = self._initmetrics()
        self.metrics =[
            smputils.metrics.IoU(),
            smputils.metrics.Fscore(),
            smputils.metrics.Recall(),
            smputils.metrics.Precision(),]
        # print(self.metrics)

        # training setting
        self.cut_type = self.traincfg['sslset']['cuttype'] \
            if 'cuttype' in self.traincfg['sslset'] \
            else None
        self.consistencyratio = self.traincfg['sslset']['consistencyratio']
        self.automatic_optimization = False

        #loss record
        #--evaluation record
        self.evaRecords = []
        self.b_counts = []
        self.loss_record_epoch = []
        self.loss_record_steps = []
        self.evaRecords_init()

    def forward(self, x, step=1):
        # in lightning, forward defines the prediction/inference actions
        """
        Args:
            x       input tensor
            step    predict branch
        """
        y_pred = getattr(self, f'branch{step}')(x)
        y_pred = torch.argmax(y_pred, dim=1)

        return y_pred

    # setting total step
    def setup(self, stage):
        if stage == 'fit':
            total_devices = self.trainer.num_devices
            self.accumulate_grad_batches = self.traincfg['traindl']['accumulate_grad_batches'] \
                * 8 // total_devices
#             self.accumulate_grad_batches = self.traincfg['traindl']['accumulate_grad_batches']
            max_epochs = self.trainer.max_epochs
            # supervised steps
            if 'sup' in self.traincfg['sslset']['type']:
                self.steps_per_epoch = len(self.trainer.datamodule.train_label_dataset)\
                     // (total_devices * self.traincfg['traindl']['batchsize'])
                self.train_steps = (max_epochs * self.steps_per_epoch) // \
                    self.accumulate_grad_batches

            # semi-supervised steps
            else:
                self.steps_per_epoch = max(
                            len(self.trainer.datamodule.train_label_dataset) //           \
                            (total_devices * self.trainer.datamodule.label_batchsize),    \
                            len(self.trainer.datamodule.train_unlabel_dataset) //         \
                            (total_devices * self.trainer.datamodule.unlabel_batchsize)
                        )
                self.train_steps = max_epochs * self.steps_per_epoch // self.accumulate_grad_batches

    def _returnCutMask(self, img_size, batch_size, lrscale=None, cut_type="cut"):
        if cut_type == "token":
            for image_i in range(batch_size):
                if image_i == 0:
                    Mask = torch.from_numpy(transformmasks.generate_tokenout_mask(img_size, lrscale)).unsqueeze(0).float()
                else:
                    Mask = torch.cat((Mask, torch.from_numpy(transformmasks.generate_tokenout_mask(img_size, lrscale)).unsqueeze(0).float()))

        elif cut_type == "cut":
            for image_i in range(batch_size):
                if image_i == 0:
                    Mask = torch.from_numpy(transformmasks.generate_cutout_mask(img_size)).unsqueeze(0).float()
                else:
                    Mask = torch.cat((Mask, torch.from_numpy(transformmasks.generate_cutout_mask(img_size)).unsqueeze(0).float()))

        return Mask.to(self.device)

    def _strongTransform(self, parameters, data=None, lrdata=None, target=None,
                        cutaug = transformsgpu.cutout, coloraug = transformsgpu.colorJitter, flipaug = transformsgpu.flip,
                        isaugsym=True):
        assert ((data is not None) or (target is not None))
        data, target = cutaug(mask = parameters.get("Cut", None), data = data, target = target)
        lrdata, _ = cutaug(mask = parameters.get("LRCut", None), data = lrdata)
        data, lrdata, target = coloraug(
            colorJitter=parameters["ColorJitter"], data=data, lrdata=lrdata, target=target,
            Threshold=0.4,saturation=0.04, hue=0.08, issymetric=isaugsym
            )
        data, lrdata, target = flipaug(flip=parameters["flip"], data=data, lrdata=lrdata, target=target)

        if not (lrdata is None):
            return data, lrdata, target
        else:
            return data, target

    def _training_sch_on_step(self):
        if self.sch_on_step == True:
            schs = self.lr_schedulers()
            if isinstance(schs, list):
                for sch in schs:
                    self._schstep(sch)
            elif schs is not None:
                self._schstep(schs)

    if pl.__version__[0] == "2":
        def on_train_epoch_end(self, outputs):
            if self.sch_on_step == False:
                schs = self.lr_schedulers()
                if isinstance(schs, list):
                    for sch in schs:
                        self._schstep(sch)
                    
                elif schs is not None:
                    self._schstep(schs)
    else:
        def training_epoch_end(self, outputs):
            if self.sch_on_step == False:
                schs = self.lr_schedulers()
                if isinstance(schs, list):
                    for sch in schs:
                        self._schstep(sch)
                    
                elif schs is not None:
                    self._schstep(schs)

    def _schstep(self, scheduler):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(self.trainer.callback_metrics["loss"])
        else:
            scheduler.step()
    
    @torch.no_grad()
    def _evaluate(self, predmask, y, stage:str):
        sync_dist = False if stage == "train" else True
        myResults = []
        if len(predmask) == 1:
            for metric_fn in self.metrics:
                metric_value = metric_fn(predmask[0], y)
                # self.log(f'{stage} {metric_fn.__name__}', metric_value, sync_dist=sync_dist)
                # myResults[metric_fn.__name__]=metric_value
                myResults.append(metric_value.item())
                if stage == "train":
                    break
                
        else:
            for metric_fn in self.metrics:
                shape = y.shape[0]
                metric_value = []
                for idx, _predmask in enumerate(predmask):
                    metric_value.append(metric_fn(_predmask[0:shape], y))
                    # self.log(f'{stage} {idx+1} {metric_fn.__name__}', metric_value[idx], sync_dist=sync_dist)
                    myResults.append(metric_value[idx].item())

                # self.log(f'{stage} {metric_fn.__name__}', \
                #     torch.tensor(metric_value).mean(), sync_dist=sync_dist)
                # myResults[metric_fn.__name__]=metric_value.mean()
            
        return myResults

    def _initloss(self):
        """
        loss initial

        Type: 
        CrossEntropyLoss
        """
        loss_type = self.traincfg['loss'].pop('type')
        if loss_type == 'CrossEntropyLoss':
            loss = nn.CrossEntropyLoss()
        else:
            raise ValueError("Loss function mismatch.")
        
        return loss

    def _initmodel(self, modelcfg):
        """
        model initial

        Type: 
        DeepLabV3Plus, Unet, UnetPlusPlus, UNeXt, SegFormer-b0,b1,b2

        Ref:
        https://smp.readthedocs.io/
        https://github.com/jeya-maria-jose/UNeXt-pytorch
        https://huggingface.co/docs/transformers/model_doc/segformer
        """
        print(modelcfg['model_seed'])
        torch.manual_seed(modelcfg['model_seed'])

        model_type = modelcfg['model'].pop('type')

        if hasattr(networks, model_type):
            model = getattr(networks, model_type)(
                **modelcfg['model']
                )
        elif hasattr(smp, model_type):
            model = getattr(smp, model_type)(
                encoder_weights = "imagenet",
                **modelcfg['model']
                )
        else:
            raise ValueError(f"Model type '{model_type}' mismatch.")
                
        
        return model

    def _initmetrics(self):
        return [
            smputils.metrics.IoU(),
            smputils.metrics.Fscore(),
            smputils.metrics.Recall(),
            smputils.metrics.Precision(),
        ]

    def _initoptimizer(self, parameters):
        """
        optimizer initial

        Type: 
        Adam, AdamW, SGD
        """
        optimcfg = copy.deepcopy(self.traincfg['optim'])
        optim_type = optimcfg.pop('type')
        if 'Adam' in optim_type and self.traincfg['expset']['precision'] == 16:
            optimcfg['eps'] = 1e-4

        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(parameters,
                                        amsgrad=True,
                                        **optimcfg)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(parameters,
                                        amsgrad=True,
                                        **optimcfg)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(parameters,
                                        momentum=0.9,
                                        weight_decay=0.0001,
                                        nesterov=True,
                                        **optimcfg
                                        )
        return optimizer

    def _initscheduler(self, optimizer):
        """
        scheduler initial

        Type: 
        CosineAnnealingWarmRestarts
        """
        # scheduler initial
        schedcfg = copy.deepcopy(self.traincfg['sched'])
        sched_type = schedcfg.pop('type')
        if sched_type == 'CosineAnnealingWR':
            self.sch_on_step = False
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                            optimizer,
                            **schedcfg
                        )
        elif sched_type == 'CosineDecayWarmUp':
            self.sch_on_step = True
            scheduler = get_cosine_schedule_with_warmup(
                            optimizer,
                            num_warmup_steps=0,
                            num_training_steps=self.train_steps,
                            **schedcfg
                        )
        elif sched_type == 'PolyDecayWarmUp':
            self.sch_on_step = True
            scheduler = get_polynomial_decay_schedule_with_warmup(
                            optimizer,
                            num_warmup_steps=0,
                            num_training_steps=self.train_steps,
                            power=0.9,
                            **schedcfg
                        )

        return scheduler

    def sigmoid_rampup(self, current, rampup_length):
        """ Exponential rampup from https://arxiv.org/abs/1610.02242 . 
        """
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def unflatten_json(self, json):
        if type(json) == dict:
            for k in sorted(json.keys(), reverse=True):
                if "." in k:
                    key_parts = k.split(".")
                    json1 = json
                    for i in range(0, len(key_parts)-1):
                        k1 = key_parts[i]
                        if k1 in json1:
                            json1 = json1[k1]
                            if type(json1) != dict:
                                conflicting_key = ".".join(key_parts[0:i+1])
                                raise Exception('Key "{}" conflicts with key "{}"'.format(
                                    k, conflicting_key))
                        else:
                            json2 = dict()
                            json1[k1] = json2
                            json1 = json2
                    if type(json1) == dict:
                        v = json.pop(k)
                        json1[key_parts[-1]] = v


    #-- evaluation record
    def evaRecords_init(self):
        self.evaRecords = []
        self.b_counts = []

    def evaRecords_append(self, records):
        self.evaRecords.append(records)

    def evaRecords_load(self):
        mean_batch = np.array(self.evaRecords).astype('float64')
        b_counts = np.array(self.b_counts).astype('float64')
        for i in range(len(b_counts)):
            mean_batch[i]/=b_counts[i]
        return mean_batch
        #return self.evaRecords



    def load_modelcfg(self, cfgpath, custom_dir=''):
        """
        Load config from traincfg.yaml 
        """
        # custom_dir ="" # modify
        with open(cfgpath, 'r') as fp:
            traincfg = yaml.load(fp, Loader=yaml.FullLoader)
    
        traincfg['modelname'] = 'MRCPS'

        if 'cps' in traincfg['sslset']['type'] \
            and 'branch2' not in traincfg:
            traincfg['branch2'] = traincfg['branch1']

        #load branch model yaml (add info in model.config)
        with open(custom_dir+traincfg['branch1'], 'r') as fp: #modify
            modelcfg = yaml.load(fp, Loader=yaml.FullLoader)
        modelcfg['model.classes'] = len(traincfg['classes'])                            # class num
        modelcfg['model_seed'] = traincfg['expset']['model_seed']                       # model random seed
        if 'model.lrscale' in modelcfg and traincfg['expset']['lrratio'] is not None:   # multiscale size
            modelcfg['model.lrscale'] = traincfg['expset']['lrratio']
        traincfg['branch1'] = modelcfg                                                  #set branch modelcfg 
        traincfg['modelnum'] = 1

        if 'branch2' in traincfg:
            with open(custom_dir+traincfg['branch2'], 'r') as fp: #modify
                modelcfg = yaml.load(fp, Loader=yaml.FullLoader)
            modelcfg['model.classes'] = len(traincfg['classes'])
            modelcfg['model_seed'] = 2 * traincfg['expset']['model_seed']
            if 'model.lrscale' in modelcfg and traincfg['expset']['lrratio'] is not None:
                modelcfg['model.lrscale'] = traincfg['expset']['lrratio']
            traincfg['branch2'] = modelcfg
            traincfg['modelnum'] = 2
            
        if 'branch3' in traincfg:
            with open(custom_dir+traincfg['branch3'], 'r') as fp: #modify
                modelcfg = yaml.load(fp, Loader=yaml.FullLoader)
            modelcfg['model.classes'] = len(traincfg['classes'])
            modelcfg['model_seed'] = traincfg['expset']['model_seed']
            if 'model.lrscale' in modelcfg and traincfg['expset']['lrratio'] is not None:
                modelcfg['model.lrscale'] = traincfg['expset']['lrratio']
            traincfg['branch3'] = modelcfg
            traincfg['modelnum'] = 3
        

        # expname
        note = traincfg['expname']
        exp_seed = traincfg['expset']['exp_seed']
        epochs = traincfg['expset']['epochs']
        traincfg['expname'] = traincfg['modelname'] + traincfg['sslset']['type'] + '_' \
                            + f"_sd{exp_seed}_e{epochs}"
        
        if 'sda' in traincfg['traindl'] and traincfg['traindl']['sda'] == True:
            traincfg['expname'] += '_s'

        if note != "":
            traincfg['expname'] += f"-{note}"

        # root path setting
        fold = traincfg['expset']['fold']
        if 'fversion' in traincfg['expset'].keys():
            fold = f"{fold}_v{traincfg['expset']['fversion']}"

        return traincfg