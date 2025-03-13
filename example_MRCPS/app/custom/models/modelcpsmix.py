import torch

import random
from .modelcps import CPSModel

class CPSMixModel(CPSModel):
    def __init__(self, traincfg):
        super().__init__(traincfg)

    def training_step(self, batch, batch_idx):
        if "label" in batch:
            x, y = batch["label"]
            predmask = []
            opt1, opt2 = self.optimizers()

            # supervised
            y_pred_1_sup = self.branch1(x)
            y_pred_2_sup = self.branch2(x)
            
            sup_loss_1 = self.criterion(y_pred_1_sup, y)
            sup_loss_2 = self.criterion(y_pred_2_sup, y)
            self.log(f"train 1 sup loss", sup_loss_1)
            self.log(f"train 2 sup loss", sup_loss_2)
            totalloss = sup_loss_1 + sup_loss_2

            predmask.append(torch.argmax(y_pred_1_sup, dim=1))
            predmask.append(torch.argmax(y_pred_2_sup, dim=1))

        if "unlabel" in batch:
            x_un = batch["unlabel"]
            with torch.no_grad():
                y_pred_un_1 = self.branch1(x_un)
                y_pred_un_2 = self.branch2(x_un)
                pseudomask_un_1 = torch.argmax(y_pred_un_1, dim=1)
                pseudomask_un_2 = torch.argmax(y_pred_un_2, dim=1)

                pseudomask_cat = torch.cat(\
                    (torch.unsqueeze(pseudomask_un_1, dim=1), torch.unsqueeze(pseudomask_un_2, dim=1)), dim=1)
                # MixMask = self._returnCutMask(pseudomask_cat.shape[2:4], pseudomask_cat.shape[0])
                strong_parameters = {}
                strong_parameters["flip"] = random.randint(0, 7)
                strong_parameters["ColorJitter"] = random.uniform(0, 1)

                mix_un_img, mix_un_mask = self._strongTransform(
                                                    strong_parameters,
                                                    data=x_un,
                                                    target=pseudomask_cat
                                                    )
                mix_un_mask_1 = torch.squeeze(mix_un_mask[:, 0:1], dim=1).long()
                mix_un_mask_2 = torch.squeeze(mix_un_mask[:, 1:2], dim=1).long()
            mix_pred_1 = self.branch1(mix_un_img)
            mix_pred_2 = self.branch2(mix_un_img)
            cps_loss_1 = self.criterion(mix_pred_1, mix_un_mask_2) * self.consistencyratio
            cps_loss_2 = self.criterion(mix_pred_2, mix_un_mask_1) * self.consistencyratio
            
            self.log(f"train 1 cps loss", cps_loss_1)
            self.log(f"train 2 cps loss", cps_loss_2)
            totalloss += (cps_loss_1 + cps_loss_2)

        self.log(f"train loss", totalloss.item() / 2, prog_bar=True)
        
        self._evaluate(predmask, y, "train")

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
