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
        self.metrics = self._initmetrics()
        # print(self.metrics)

        # training setting
        self.cut_type = self.traincfg['sslset']['cuttype'] \
            if 'cuttype' in self.traincfg['sslset'] \
            else None
        self.consistencyratio = self.traincfg['sslset']['consistencyratio']
        self.automatic_optimization = False

        #loss record
        self.loss_record_epoch = []
        self.loss_record_steps = []
        
        
        #model bulid
        self.unflatten_json(self.traincfg['branch1'])
        self.branch1 = self._initmodel(self.traincfg['branch1'])

        self.unflatten_json(self.traincfg['branch2'])
        self.branch2 = self._initmodel(self.traincfg['branch2'])


    def forward(self, x, lrx, step=1):
        # in lightning, forward defines the prediction/inference actions
        """
        Args:
            x       input tensor
            step    predict branch
        """
        if step == 0:
            return torch.argmax(
                self.branch1(x, lrx).softmax(1) + self.branch2(x, lrx).softmax(1)
                , dim=1)
        elif step == 3:
            p1 = self.branch1(x, lrx).softmax(1)
            p2 = self.branch2(x, lrx).softmax(1)
            return torch.argmax(p1+p2, dim=1), torch.argmax(p1, dim=1), torch.argmax(p2, dim=1)
        else:
            return torch.argmax(getattr(self, f'branch{step}')(x, lrx), dim=1)
    def test_step(self, batch, batch_idx):
        # return
        image, mask, lrimage = batch
        predmask = []
        y_pred_1 = self.branch1(image, lrimage)
        predmask.append(torch.argmax(y_pred_1, dim=1))
        y_pred_2 = self.branch2(image, lrimage)
        predmask.append(torch.argmax(y_pred_2, dim=1))
        
        result =self._evaluate(predmask, mask, "test")

        predensem = []
        voting = y_pred_1.softmax(1) + y_pred_2.softmax(1)
        predensem.append(torch.argmax(voting, dim=1))

        result_ens = self._evaluate(predensem, mask, "test ens")

    def configure_optimizers(self):
        opts = []
        schs = []

        optimizer1 = self._initoptimizer(self.branch1.parameters())
        scheduler1 = self._initscheduler(optimizer1)
        opts.append(optimizer1)
        schs.append(scheduler1)

        optimizer2 = self._initoptimizer(self.branch2.parameters())
        scheduler2 = self._initscheduler(optimizer2)
        opts.append(optimizer2)
        schs.append(scheduler2)

        return opts, schs


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
            self.epoch_loss()
            if self.sch_on_step == False:
                schs = self.lr_schedulers()
                if isinstance(schs, list):
                    for sch in schs:
                        self._schstep(sch)
                    
                elif schs is not None:
                    self._schstep(schs)
    else:
        def training_epoch_end(self, outputs):
            self.epoch_loss()
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
                self.log(f'{stage} {metric_fn.__name__}', metric_value, sync_dist=sync_dist)
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
                    self.log(f'{stage} {idx+1} {metric_fn.__name__}', metric_value[idx], sync_dist=sync_dist)
                    myResults.append(metric_value[idx].item())

                self.log(f'{stage} {metric_fn.__name__}', \
                    torch.tensor(metric_value).mean(), sync_dist=sync_dist)
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
    
    
#---MRCPS mix training--
    def training_step(self, batch, batch_idx):
        # print('--------------------------------')
        # print(batch.keys())
        opt1, opt2 = self.optimizers()
        # lose_dict = {'total':0}
        self.loss_record_df = pd.DataFrame(columns=["b1_sup", "b2_sup", "b1_cps", "b2_cps", "total"])
        self.loss_record_epoch_df = pd.DataFrame(columns=["b1_sup", "b2_sup", "b1_cps", "b2_cps", "total"])
        lose_dict = {'b1_sup': 0, 'b2_sup': 0, 'b1_cps': 0, 'b2_cps': 0, 'total': 0}

        if "label" in batch:
            image, mask, lrimage = batch["label"]
            # print('--------------------------------')
            # print(image.shape)

            if len(image)>0:
                predmask = []

                # supervised
                y_pred_1_sup = self.branch1(image, lrimage)
                y_pred_2_sup = self.branch2(image, lrimage)
                
                sup_loss_1 = self.criterion(y_pred_1_sup, mask)
                sup_loss_2 = self.criterion(y_pred_2_sup, mask)
                self.log(f"train 1 sup loss", sup_loss_1)
                self.log(f"train 2 sup loss", sup_loss_2)
                totalloss = sup_loss_1 + sup_loss_2

                predmask.append(torch.argmax(y_pred_1_sup, dim=1))
                predmask.append(torch.argmax(y_pred_2_sup, dim=1))

                lose_dict['b1_sup'] = sup_loss_1.item()
                lose_dict['b2_sup'] = sup_loss_2.item()
                lose_dict['total'] += sup_loss_1.item()+sup_loss_2.item()

        if "unlabel" in batch:
            image_un, lrimage_un = batch["unlabel"]
            # print('--------------------------------')
            # print(image_un.shape)

            with torch.no_grad():
                y_pred_un_1 = self.branch1(image_un, lrimage_un)
                y_pred_un_2 = self.branch2(image_un, lrimage_un)
                pseudomask_un_1 = torch.argmax(y_pred_un_1, dim=1)
                pseudomask_un_2 = torch.argmax(y_pred_un_2, dim=1)

                pseudomask_cat = torch.cat(\
                    (torch.unsqueeze(pseudomask_un_1, dim=1), torch.unsqueeze(pseudomask_un_2, dim=1)), dim=1)
                strong_parameters = {}
                if self.traincfg['sslset'].get("iscut", False):
                    if random.uniform(0, 1) < 0.5:
                        MixMask = self._returnCutMask(pseudomask_cat.shape[2:4], pseudomask_cat.shape[0], cut_type="cut")
                        strong_parameters = {"Cut": MixMask}
                strong_parameters["flip"] = random.randint(0, 7)
                strong_parameters["ColorJitter"] = random.uniform(0, 1)

                mix_un_img, mix_un_lrimg, mix_un_mask = self._strongTransform(
                                                    strong_parameters,
                                                    data=image_un,
                                                    lrdata=lrimage_un,
                                                    target=pseudomask_cat,
                                                    isaugsym=self.traincfg['sslset'].get("isaugsym", True)
                                                    )

                mix_un_mask_1 = torch.squeeze(mix_un_mask[:, 0:1], dim=1).long()
                mix_un_mask_2 = torch.squeeze(mix_un_mask[:, 1:2], dim=1).long()
                mix_pred_1 = self.branch1(mix_un_img, mix_un_lrimg)
                mix_pred_2 = self.branch2(mix_un_img, mix_un_lrimg)
                cps_loss_1 = self.criterion(mix_pred_1, mix_un_mask_2) * self.consistencyratio
                cps_loss_2 = self.criterion(mix_pred_2, mix_un_mask_1) * self.consistencyratio
                
                self.log(f"train 1 cps loss", cps_loss_1)
                self.log(f"train 2 cps loss", cps_loss_2)
                totalloss += (cps_loss_1 + cps_loss_2)

                
                lose_dict['b1_cps'] = cps_loss_1.item()
                lose_dict['b2_cps'] = cps_loss_2.item()
                lose_dict['total'] += sup_loss_1.item()+sup_loss_2.item()

        self.log(f"train loss", totalloss.item() / 2, prog_bar=True)
        
        if "label" in batch:
            self._evaluate(predmask, mask, "train")

        # backwarding
        totalloss /= self.accumulate_grad_batches
        self.manual_backward(totalloss)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt1.step()
            opt2.step()
            opt1.zero_grad()
            opt2.zero_grad()
        
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self._training_sch_on_step()

        self.loss_record_steps.append([lose_dict['b1_sup'],lose_dict['b2_sup'],
                                       lose_dict['b1_cps'],lose_dict['b2_cps'],
                                       lose_dict['total']])
        
    def epoch_loss(self):
        loss_array = np.array(self.loss_record_steps)
        loss_avg = np.average(loss_array, axis=0)
        self.loss_record_epoch.append(loss_avg)
        self.loss_record_steps = []
        
