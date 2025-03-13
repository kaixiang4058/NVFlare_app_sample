import torch
import random

from .modelmrcps import MRCPSModel

class MSCSDualModel(MRCPSModel):
    def training_step(self, batch, batch_idx):
        image, mask, lrimage = batch["label"]

        predmask = []
        opt1, opt2 = self.optimizers()

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

        if "unlabel" in batch:
            image_un, lrimage_un = batch["unlabel"]

            y_pred_un_1 = self.branch1(image_un, lrimage_un)
            y_pred_un_2 = self.branch2(image_un, lrimage_un)
            with torch.no_grad():
                pseudomask_un_1 = torch.argmax(y_pred_un_1, dim=1)
                pseudomask_un_2 = torch.argmax(y_pred_un_2, dim=1)

                pseudomask_cat = torch.cat(\
                    (torch.unsqueeze(pseudomask_un_1, dim=1), torch.unsqueeze(pseudomask_un_2, dim=1)), dim=1)
                strong_parameters = {}
                strong_parameters["flip"] = random.randint(0, 7)
                strong_parameters["ColorJitter"] = 1    # Always apply

                # First image perturbation
                mix_un_img, mix_un_lrimg, mix_un_mask = self._strongTransform(
                                                    strong_parameters,
                                                    data=image_un,
                                                    lrdata=lrimage_un,
                                                    target=pseudomask_cat
                                                    )
                mix_un_mask_1 = torch.squeeze(mix_un_mask[:, 0:1], dim=1).long()
                mix_un_mask_2 = torch.squeeze(mix_un_mask[:, 1:2], dim=1).long()

            mix_pred_1 = self.branch1(mix_un_img, mix_un_lrimg)
            mix_pred_2 = self.branch2(mix_un_img, mix_un_lrimg)

            total_pred_1 = torch.cat((y_pred_un_1, mix_pred_1))
            total_pred_2 = torch.cat((y_pred_un_2, mix_pred_2))
            total_mask_1 = torch.cat((pseudomask_un_1, mix_un_mask_1))
            total_mask_2 = torch.cat((pseudomask_un_2, mix_un_mask_2))

            cps_loss_1 = self.criterion(total_pred_1, total_mask_2) * self.consistencyratio
            cps_loss_2 = self.criterion(total_pred_2, total_mask_1) * self.consistencyratio
            
            self.log(f"train 1 cps loss", cps_loss_1)
            self.log(f"train 2 cps loss", cps_loss_2)
            totalloss += (cps_loss_1 + cps_loss_2)

        self.log(f"train loss", totalloss.item() / 2, prog_bar=True)
        
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
