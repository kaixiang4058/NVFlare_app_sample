import torch
import copy
import segmentation_models_pytorch.utils as smputils

from .modelbase import BaseModel
import torchvision.transforms.functional as F
import random

class MLCModel(BaseModel):
    def __init__(self, traincfg):
        super().__init__(traincfg)

        self.unflatten_json(self.traincfg['branch1'])
        self.branch1 = self._initmodel(self.traincfg['branch1'])

        self.unflatten_json(self.traincfg['branch2'])
        self.branch2 = self._initmodel(self.traincfg['branch2'])

        self.unflatten_json(self.traincfg['branch3'])
        self.branch3 = self._initmodel(self.traincfg['branch3'])

        self.evaRecord={}
    def forward(self, x, lrx, step=1):
        # in lightning, forward defines the prediction/inference actions
        """
        Args:
            x       input tensor
            step    predict branch
        """
        if step == 0:
            return torch.argmax(
                self.branch1(x).softmax(1) + self.branch2(x).softmax(1)
                , dim=1)
        elif step == 3:
            p1 = self.branch1(x).softmax(1)
            p2 = self.branch2(x).softmax(1)
            return torch.argmax(p1+p2, dim=1), torch.argmax(p1, dim=1), torch.argmax(p2, dim=1)
        else:
            return torch.argmax(getattr(self, f'branch{step}')(x), dim=1)

    def training_step(self, batch, batch_idx):
        # image, mask, lrimage, lrmask = batch["label"]
        # lrmask = self._downpool(lrmask)
        totalloss=0
        opt1, opt2, opt3 = self.optimizers()

        #---supervised---
        if "label" in batch:
            predmask = []
            image, mask, lrimage, lrmask = batch["label"]

            #cell level predict
            sup_pred_1 = self.branch1(image)
            sup_pred_2 = self.branch2(image)
            #tissue level predict
            sup_pred_3, sup_pred_3_centroid = self.branch3(lrimage)
            #get mask from tissue level centroid
            sup_pseudomask_3_centroid = torch.argmax(sup_pred_3_centroid, dim=1)
            #resize for scale consistency
            sup_pred_1_resize = F.resize(sup_pred_1, sup_pseudomask_3_centroid.shape[2])#
            sup_pred_2_resize = F.resize(sup_pred_2, sup_pseudomask_3_centroid.shape[2])

            #---loss---
            #supervised loss compute
            sup_loss_1 = self.criterion(sup_pred_1, mask)
            sup_loss_2 = self.criterion(sup_pred_2, mask)
            sup_loss_3 = self.criterion(sup_pred_3, lrmask)
            
            #multi scale consistency loss
            sup_multscaleloss_1 = self.criterion(sup_pred_1_resize, sup_pseudomask_3_centroid)
            sup_multscaleloss_2 = self.criterion(sup_pred_2_resize, sup_pseudomask_3_centroid)


            #loss log
            self.log(f"train 1(cell) sup loss", sup_loss_1)
            self.log(f"train 2(cell) sup loss", sup_loss_2)
            self.log(f"train 3(tissue) sup loss", sup_loss_3)
            self.log(f"train 1(cell),3(tissue) sup consistency loss", sup_multscaleloss_1)
            self.log(f"train 2(cell),3(tissue) sup consistency loss", sup_multscaleloss_2)
            totalloss += sup_loss_1 + sup_loss_2 + sup_multscaleloss_1 + sup_multscaleloss_2 + sup_loss_3

            #record predict mask(cell level)
            predmask.append(torch.argmax(sup_pred_1, dim=1))
            predmask.append(torch.argmax(sup_pred_2, dim=1))

        #---unsupersized---
        if "unlabel" in batch:
            image_un, lrimage_un = batch["unlabel"]
            with torch.no_grad():
                #cell level prdict for mask (no grad)
                unsup_pred_1 = self.branch1(image_un)
                unsup_pred_2 = self.branch2(image_un)
                unsup_pseudomask_1 = torch.argmax(unsup_pred_1, dim=1)
                unsup_pseudomask_2 = torch.argmax(unsup_pred_2, dim=1)
                #tissue level prdict for mask (no grad)
                unsup_pred_3, unsup_pred_3_centroid= self.branch3(image_un)
                unsup_pseudomask_3 = torch.argmax(unsup_pred_3, dim=1)

                #mask concat for augmentation
                pseudomask_cat = torch.cat(\
                    (torch.unsqueeze(unsup_pseudomask_1, dim=1), torch.unsqueeze(unsup_pseudomask_2, dim=1), torch.unsqueeze(unsup_pseudomask_3, dim=1)), dim=1)
                
                #agumentation setting
                strong_parameters = {}
                # if random.uniform(0, 1) < 0.5:
                #     MixMask = self._returnCutMask(pseudomask_cat.shape[2:4], pseudomask_cat.shape[0], cut_type="cut")
                #     strong_parameters = {"Cut": MixMask}
                strong_parameters["flip"] = random.randint(0, 7)
                strong_parameters["ColorJitter"] = random.uniform(0, 1)
                
                #augmentation
                unsup_miximg, unsup_mixlrimg, unsup_mixmask = self._strongTransform(
                                                    strong_parameters,
                                                    data=image_un,
                                                    lrdata=lrimage_un,
                                                    target=pseudomask_cat,
                                                    isaugsym=self.traincfg['sslset'].get("isaugsym", True)
                                                    )

                #get augmentation mask
                unsup_augmask_1 = torch.squeeze(unsup_mixmask[:, 0:1], dim=1).long()
                unsup_augmask_2 = torch.squeeze(unsup_mixmask[:, 1:2], dim=1).long()
                unsup_augmask_3 = torch.squeeze(unsup_mixmask[:, 2:3], dim=1).long()

                #resize for multi scale consistency 
                unsup_augmask_1_resize = F.resize(unsup_augmask_1, unsup_pred_3_centroid.shape[2])#
                unsup_augmask_2_resize = F.resize(unsup_augmask_2, unsup_pred_3_centroid.shape[2])


            #augmentation predict (with grad)
            unsup_augpred_1 = self.branch1(unsup_miximg)
            unsup_augpred_2 = self.branch2(unsup_miximg)
            unsup_augpred_3, unsup_augpred_3_centroid = self.branch3(unsup_mixlrimg)
            # unsup_augpred_1_resize = F.resize(unsup_augpred_1, unsup_pred_3_centroid.shape[2])
            # unsup_augpred_2_resize = F.resize(unsup_augpred_2, unsup_pred_3_centroid.shape[2])


            #---loss---
            #augmentatino loss of tissue level
            unsup_aug_loss_3 = self.criterion(unsup_augpred_3, unsup_augmask_3)

            #multi scale + augmentation loss
            # unsup_multscale_loss_p1m3 = self.criterion(unsup_augpred_1_resize, unsup_augmask_3)
            # unsup_multscale_loss_p2m3 = self.criterion(unsup_augpred_2_resize, unsup_augmask_3)

            unsup_multscale_loss_p3m1 = self.criterion(unsup_augpred_3_centroid, unsup_augmask_1_resize)
            unsup_multscale_loss_p3m2 = self.criterion(unsup_augpred_3_centroid, unsup_augmask_2_resize)

            #multi model + augmentation loss
            unsup_difmodel_loss_1 = self.criterion(unsup_augpred_1, unsup_augmask_2)
            unsup_difmodel_loss_2 = self.criterion(unsup_augpred_2, unsup_augmask_1)
            
            # self.log(f"train 1(cell),3(tissue) sup consistency loss", unsup_multscale_loss_p1m3)
            # self.log(f"train 1(cell),3(tissue) sup consistency loss", unsup_multscale_loss_p2m3)
            self.log(f"train 3 augment loss", unsup_aug_loss_3)
            self.log(f"train predict3(tissue),mask1(cell) sup consistency loss", unsup_multscale_loss_p3m1)
            self.log(f"train predict3(tissue),mask2(cell) sup consistency loss", unsup_multscale_loss_p3m2)
            self.log(f"train 1 different model loss", unsup_difmodel_loss_1)
            self.log(f"train 2 different model loss", unsup_difmodel_loss_2)
            totalloss += (unsup_difmodel_loss_1 + unsup_difmodel_loss_2\
                            # +unsup_multscale_loss_p1m3+unsup_multscale_loss_p2m3\
                            +unsup_multscale_loss_p3m1+unsup_multscale_loss_p3m2\
                            +unsup_aug_loss_3\
                            )* self.consistencyratio


        self.log(f"train loss", totalloss.item() / 2, prog_bar=True)
        
        self._evaluate(predmask, mask, "train")

        # backwarding
        totalloss /= self.accumulate_grad_batches
        self.manual_backward(totalloss)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt1.step()
            opt2.step()
            opt3.step()
            opt1.zero_grad()
            opt2.zero_grad()
            opt3.zero_grad()
        
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self._training_sch_on_step()

    def validation_step(self, batch, batch_idx):
        # return
        image, mask, lrimage = batch
        predmask = []
        y_pred_1 = self.branch1(image)
        predmask.append(torch.argmax(y_pred_1, dim=1))
        y_pred_2 = self.branch2(image)
        predmask.append(torch.argmax(y_pred_2, dim=1))

        loss_1 = self.criterion(y_pred_1, mask)
        self.log(f"valid 1 loss", loss_1)
        loss_2 = self.criterion(y_pred_2, mask)
        self.log(f"valid 2 loss", loss_2)

        self.log(f"valid loss", (loss_1 + loss_2) / 2, prog_bar=True)

        eva_score = self._evaluate(predmask, mask, "valid")
        self.evalution_record([loss_1.item(), eva_score[0]], task='valid b1')
        self.evalution_record([loss_2.item(), eva_score[1]], task='valid b2')

        predensem = []
        voting = y_pred_1.softmax(1) + y_pred_2.softmax(1)
        predensem.append(torch.argmax(voting, dim=1))

        eva_score =self._evaluate(predensem, mask, "valid ens")
        ens_loss = (loss_1 + loss_2)/2
        self.evalution_record([ens_loss.item(), eva_score[0]], task='valid ens')

    def test_step(self, batch, batch_idx):
        # return
        image, mask, lrimage = batch
        predmask = []
        y_pred_1 = self.branch1(image)
        predmask.append(torch.argmax(y_pred_1, dim=1))
        y_pred_2 = self.branch2(image)
        predmask.append(torch.argmax(y_pred_2, dim=1))
        
        self._evaluate(predmask, mask, "test")

        predensem = []
        voting = y_pred_1.softmax(1) + y_pred_2.softmax(1)
        predensem.append(torch.argmax(voting, dim=1))

        self._evaluate(predensem, mask, "test ens")

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

        optimizer3 = self._initoptimizer(self.branch3.parameters())
        scheduler3 = self._initscheduler(optimizer3)
        opts.append(optimizer3)
        schs.append(scheduler3)

        return opts, schs

    def _initmetrics(self):
        return [
            smputils.metrics.IoU(),
        ]
        

    def evalution_record(self, recordinfo, task):
        if task not in self.evaRecord.keys():
            self.evaRecord[task]=[]
        self.evaRecord[task].append(recordinfo)
