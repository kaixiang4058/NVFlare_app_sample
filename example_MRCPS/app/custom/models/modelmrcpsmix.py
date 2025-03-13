import torch
import random
import numpy as np
from .modelmrcps import MRCPSModel

class MRCPSMixModel(MRCPSModel):
    def training_step(self, batch, batch_idx):
        # print('--------------------------------')
        # print(batch.keys())
        opt1, opt2 = self.optimizers()
        totalloss = 0
        lose_dict = {'total':0, 'b1_sup':0, 'b2_sup':0,
                    'b1_cps':0, 'b2_cps':2}
        
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
                # self.log(f"train 1 sup loss", sup_loss_1)
                # self.log(f"train 2 sup loss", sup_loss_2)
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
            
            # self.log(f"train 1 cps loss", cps_loss_1)
            # self.log(f"train 2 cps loss", cps_loss_2)
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
        

    def on_train_epoch_end(self):
        loss_array = np.array(self.loss_record_steps)
        loss_avg = np.average(loss_array, axis=0)
        self.loss_record_epoch.append(loss_avg)
        # self.loss_record_epoch=loss_avg
        
        self.loss_record_steps = []
        if self.sch_on_step == False:
            schs = self.lr_schedulers()
            if isinstance(schs, list):
                for sch in schs:
                    self._schstep(sch)

            elif schs is not None:
                self._schstep(schs)