import torch
import copy
import segmentation_models_pytorch.utils as smputils
import random
from .modelbase import BaseModel
import torch.nn.functional as F

class MSCCSModel(BaseModel):
    def __init__(self, traincfg):
        super().__init__(traincfg)

        self.unflatten_json(self.traincfg['branch1'])
        tmp_b1 = copy.deepcopy(self.traincfg['branch1'])
        self.unflatten_json(self.traincfg['branch2'])
        tmp_b2 = copy.deepcopy(self.traincfg['branch2'])
        self.model_c_1 = self._initmodel(self.traincfg['branch1'])
        self.model_c_2 = self._initmodel(self.traincfg['branch2'])

        self.model_t_1 = self._initmodel(tmp_b1)
        self.model_t_2 = self._initmodel(tmp_b2)

        self.lrratio = self.traincfg['expset'].get('lrratio', 8)

    def forward(self, x, step=1, level='c'):
        # in lightning, forward defines the prediction/inference actions
        """
        Args:
            x       input tensor
            step    predict branch
        """
        y_pred = getattr(self, f'model_{level}_{step}')(x)
        y_pred = torch.argmax(y_pred, dim=1)

        return y_pred

    def training_step(self, batch, batch_idx):
        image, mask, lrimage, lrmask = batch["label"]

        opt1, opt2, opt3, opt4 = self.optimizers()

        # supervised
        y_pred_1_c_sup = self.model_c_1(image)
        y_pred_2_c_sup = self.model_c_2(image)
        
        sup_c_loss_1 = self.criterion(y_pred_1_c_sup, mask)
        sup_c_loss_2 = self.criterion(y_pred_2_c_sup, mask)
        self.log(f"train 1 sup loss", sup_c_loss_1)
        self.log(f"train 2 sup loss", sup_c_loss_2)
        totalloss = sup_c_loss_1 + sup_c_loss_2

        predmask = []
        predmask.append(torch.argmax(y_pred_1_c_sup, dim=1))
        predmask.append(torch.argmax(y_pred_2_c_sup, dim=1))
        self._evaluate(predmask, mask, "train")

        y_pred_1_t_sup = self.model_t_1(lrimage)
        y_pred_2_t_sup = self.model_t_2(lrimage)
        sup_t_loss_1 = self.criterion(y_pred_1_t_sup, lrmask)
        sup_t_loss_2 = self.criterion(y_pred_2_t_sup, lrmask)
        self.log(f"train 1 sup t loss", sup_t_loss_1)
        self.log(f"train 2 sup t loss", sup_t_loss_2)
        totalloss += sup_t_loss_1 + sup_t_loss_2

        predmask = []
        predmask.append(torch.argmax(y_pred_1_t_sup, dim=1))
        predmask.append(torch.argmax(y_pred_2_t_sup, dim=1))
        self._evaluate(predmask, mask, "train t")

        if "unlabel" in batch:
            image_un, lrimage_un = batch["unlabel"]
            with torch.no_grad():
                y_pred_un_c_1 = self.model_c_1(image_un)
                y_pred_un_c_2 = self.model_c_2(image_un)

                y_pred_un_t_1 = self.model_t_1(lrimage_un)
                y_pred_un_t_2 = self.model_t_2(lrimage_un)
            
                pse_un_c_1 = torch.argmax(y_pred_un_c_1, dim=1)
                pse_un_c_2 = torch.argmax(y_pred_un_c_2, dim=1)
                pse_un_c_esm = torch.argmax(y_pred_un_c_1.softmax(1) + y_pred_un_c_2.softmax(1), dim=1)

                pse_un_t_1 = torch.argmax(y_pred_un_t_1, dim=1)
                pse_un_t_2 = torch.argmax(y_pred_un_t_2, dim=1)
                pse_un_t_esm = torch.argmax(y_pred_un_t_1.softmax(1) + y_pred_un_t_2.softmax(1), dim=1)

                pse_cat = torch.cat(\
                    (torch.unsqueeze(pse_un_c_1, dim=1),
                    torch.unsqueeze(pse_un_c_2, dim=1),
                    torch.unsqueeze(pse_un_c_esm, dim=1),
                    torch.unsqueeze(pse_un_t_1, dim=1),
                    torch.unsqueeze(pse_un_t_2, dim=1),
                    torch.unsqueeze(pse_un_t_esm, dim=1)),
                    dim=1)
                
                strong_parameters = {}
                strong_parameters["flip"] = random.randint(0, 7)
                strong_parameters["ColorJitter"] = random.uniform(0, 1)

                mix_un_img, mix_un_lrimg, mix_un_mask = self._strongTransform(
                                                    strong_parameters,
                                                    data=image_un,
                                                    lrdata=lrimage_un,
                                                    target=pse_cat
                                                    )
                mix_un_c_mask_1 = torch.squeeze(mix_un_mask[:, 0:1], dim=1).long()
                mix_un_c_mask_2 = torch.squeeze(mix_un_mask[:, 1:2], dim=1).long()

                mix_un_c_mask_esb = mix_un_mask[:, 2:3].float()
                H, W = image.size()[-2:]
                mix_un_c_mask_esb = F.interpolate(
                    mix_un_c_mask_esb, size=(H//self.lrratio, W//self.lrratio), mode='area')
                mix_un_c_mask_esb = torch.squeeze(torch.round(mix_un_c_mask_esb), dim=1).long()

                mix_un_t_mask_1 = torch.squeeze(mix_un_mask[:, 3:4], dim=1).long()
                mix_un_t_mask_2 = torch.squeeze(mix_un_mask[:, 4:5], dim=1).long()

                mix_un_t_mask_esb = mix_un_mask[:, 5:6].float()
                mix_un_t_mask_esb = mix_un_t_mask_esb[..., H//2-H//2//self.lrratio-1:H//2+H//2//self.lrratio+1, \
                        W//2-W//2//self.lrratio-1:W//2+W//2//self.lrratio+1]
                mix_un_t_mask_esb = F.interpolate(mix_un_t_mask_esb, scale_factor=self.lrratio, mode="bilinear", align_corners=False)
                mix_un_t_mask_esb = mix_un_t_mask_esb[..., self.lrratio:mix_un_t_mask_esb.shape[-2]-self.lrratio,\
                                    self.lrratio:mix_un_t_mask_esb.shape[-1]-self.lrratio]
                mix_un_t_mask_esb = torch.squeeze(torch.round(mix_un_t_mask_esb), dim=1).long()
            
            mix_pred_c_1 = self.model_c_1(mix_un_img)
            mix_pred_c_2 = self.model_c_2(mix_un_img)
            mix_pred_t_1 = self.model_t_1(mix_un_lrimg)
            mix_pred_t_2 = self.model_t_2(mix_un_lrimg)

            cps_loss_c_1 = self.criterion(mix_pred_c_1, mix_un_c_mask_2) * self.consistencyratio
            cps_loss_c_2 = self.criterion(mix_pred_c_2, mix_un_c_mask_1) * self.consistencyratio
            cps_loss_t_1 = self.criterion(mix_pred_t_1, mix_un_t_mask_2) * self.consistencyratio
            cps_loss_t_2 = self.criterion(mix_pred_t_2, mix_un_t_mask_1) * self.consistencyratio

            mc_loss_c_1 = self.criterion(mix_pred_c_1, mix_un_t_mask_esb) * self.consistencyratio
            mc_loss_c_2 = self.criterion(mix_pred_c_2, mix_un_t_mask_esb) * self.consistencyratio

            mix_pred_t_1_center = mix_pred_t_1[..., H//2-H//2//self.lrratio:H//2+H//2//self.lrratio, \
                        W//2-W//2//self.lrratio:W//2+W//2//self.lrratio]
            mix_pred_t_2_center = mix_pred_t_2[..., H//2-H//2//self.lrratio:H//2+H//2//self.lrratio, \
                        W//2-W//2//self.lrratio:W//2+W//2//self.lrratio]
            mc_loss_t_1 = self.criterion(mix_pred_t_1_center, mix_un_c_mask_esb) * self.consistencyratio
            mc_loss_t_2 = self.criterion(mix_pred_t_2_center, mix_un_c_mask_esb) * self.consistencyratio
            
            self.log(f"train cps loss", cps_loss_c_1 + cps_loss_c_2)
            self.log(f"train t cps loss", cps_loss_t_1 + cps_loss_t_2)
            self.log(f"train mc loss", mc_loss_c_1 + mc_loss_c_2)
            self.log(f"train t mc loss", mc_loss_t_1 + mc_loss_t_2)

            totalloss += (cps_loss_c_1 + cps_loss_c_2 + \
                            cps_loss_t_1 + cps_loss_t_2 + \
                            mc_loss_c_1 + mc_loss_c_2 + \
                            mc_loss_t_1 + mc_loss_t_2)

        self.log(f"train loss", totalloss.item() / 2, prog_bar=True)

        # backwarding
        totalloss /= self.accumulate_grad_batches
        self.manual_backward(totalloss)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt1.step()
            opt2.step()
            opt3.step()
            opt4.step()
            opt1.zero_grad()
            opt2.zero_grad()
            opt3.zero_grad()
            opt4.zero_grad()
        
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self._training_sch_on_step()

    def validation_step(self, batch, batch_idx):
        # return
        image, mask, lrimage, lrmask = batch
        predmask = []
        y_pred_1 = self.model_c_1(image)
        predmask.append(torch.argmax(y_pred_1, dim=1))
        y_pred_2 = self.model_c_2(image)
        predmask.append(torch.argmax(y_pred_2, dim=1))

        loss_1 = self.criterion(y_pred_1, mask)
        self.log(f"valid 1 loss", loss_1)
        loss_2 = self.criterion(y_pred_2, mask)
        self.log(f"valid 2 loss", loss_2)

        self.log(f"valid loss", (loss_1 + loss_2) / 2, prog_bar=True)

        self._evaluate(predmask, mask, "valid")

        predmask = []
        y_pred_1 = self.model_t_1(lrimage)
        predmask.append(torch.argmax(y_pred_1, dim=1))
        y_pred_2 = self.model_t_2(lrimage)
        predmask.append(torch.argmax(y_pred_2, dim=1))

        loss_1 = self.criterion(y_pred_1, lrmask)
        self.log(f"valid t 1 loss", loss_1)
        loss_2 = self.criterion(y_pred_2, lrmask)
        self.log(f"valid t 2 loss", loss_2)

        self._evaluate(predmask, lrmask, "valid t")

    def test_step(self, batch, batch_idx):
        # return
        image, mask, lrimage, lrmask = batch
        predmask = []
        y_pred_1 = self.model_c_1(image)
        predmask.append(torch.argmax(y_pred_1, dim=1))
        y_pred_2 = self.model_c_2(image)
        predmask.append(torch.argmax(y_pred_2, dim=1))
        
        self._evaluate(predmask, mask, "test")

        predmask = []
        y_pred_1 = self.model_t_1(lrimage)
        predmask.append(torch.argmax(y_pred_1, dim=1))
        y_pred_2 = self.model_t_2(lrimage)
        predmask.append(torch.argmax(y_pred_2, dim=1))
        
        self._evaluate(predmask, lrmask, "test t")

    def configure_optimizers(self):
        opts = []
        schs = []

        optimizer1 = self._initoptimizer(self.model_c_1.parameters())
        scheduler1 = self._initscheduler(optimizer1)
        opts.append(optimizer1)
        schs.append(scheduler1)

        optimizer2 = self._initoptimizer(self.model_c_2.parameters())
        scheduler2 = self._initscheduler(optimizer2)
        opts.append(optimizer2)
        schs.append(scheduler2)

        optimizer3 = self._initoptimizer(self.model_t_1.parameters())
        scheduler3 = self._initscheduler(optimizer3)
        opts.append(optimizer3)
        schs.append(scheduler3)

        optimizer4 = self._initoptimizer(self.model_t_2.parameters())
        scheduler4 = self._initscheduler(optimizer4)
        opts.append(optimizer4)
        schs.append(scheduler4)

        return opts, schs

    def _initmetrics(self):
        return [
            smputils.metrics.IoU(),
        ]