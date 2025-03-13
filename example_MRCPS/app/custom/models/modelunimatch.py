import torch
import torch.nn as nn

import segmentation_models_pytorch.utils as smputils
import random
from models import transformsgpu

from models.unimatch.models.segformer import SegFormer
from models.unimatch.models.unet import Unet

from .modelbase import BaseModel

class UniMatchModel(BaseModel):
    def __init__(self, traincfg):
        super().__init__(traincfg)

        self.unflatten_json(self.traincfg['branch1'])

        model_type = self.traincfg['branch1']['model'].pop('type')
        torch.manual_seed(self.traincfg['branch1']['model_seed'])

        if model_type == "SegFormer":
            self.model = SegFormer()
        elif model_type == "Unet":
            self.model = Unet()

        # create criterions
        # TODO: support more types of the consistency criterion
        self.criterion_u = nn.CrossEntropyLoss(reduction='none')
        self.conf_thresh = 0.95
        self.warmupstep = 1000

    def forward(self, x, step=None):
        """
        Args:
            x       input tensor
            step    predict branch
        """

        y_pred = self.model(x)
        y_pred = torch.argmax(y_pred, dim=1)

        return y_pred

    def training_step(self, batch, batch_idx):
        img_x, mask_x = batch["label"]
        img_u_w = batch["unlabel"]
        b, _, h, w = img_u_w.shape
        crit_pixel = b*h*w

        predmask = []
        opt = self.optimizers()

        pred_x = self.model(img_x)
        predmask.append(torch.argmax(pred_x, dim=1))
        self._evaluate(predmask, mask_x, "train")

        loss_x = self.criterion(pred_x, mask_x)

        totalloss = loss_x

        # global_step is calculated based on optimizer step.
        if self.global_step >= self.warmupstep:
            with torch.no_grad():
                pred_u_w_mix = self.model(img_u_w).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            with torch.no_grad():       
                mask_u_w_mix_tmp = torch.unsqueeze(mask_u_w_mix, dim=1)
                conf_u_w_mix_tmp = torch.unsqueeze(conf_u_w_mix, dim=1)
                pseudomask_un_t = torch.concat((mask_u_w_mix_tmp, conf_u_w_mix_tmp), dim=1)

                # s1
                if random.uniform(0, 1) < 0.5:
                    MixMask = self._returnCutMask(pseudomask_un_t.shape[2:4], pseudomask_un_t.shape[0], cut_type="cut")
                    strong_parameters = {"Cut": MixMask}
                else:
                    strong_parameters = {"Cut": None}
                strong_parameters["flip"] = random.randint(0, 7)
                if strong_parameters["Cut"] == None:
                    strong_parameters["ColorJitter"] = 1
                else:
                    strong_parameters["ColorJitter"] = random.uniform(0, 1)

                img_u_s1, mix_un_mask_s1 = self._strongTransform(
                                                    strong_parameters,
                                                    data=img_u_w,
                                                    target=pseudomask_un_t,
                                                    cutaug=transformsgpu.mix
                                                    )
                mask_u_w_cutmixed1 = mix_un_mask_s1[:, 0:1]
                conf_u_w_cutmixed1 = mix_un_mask_s1[:, 1:2]

                mask_u_w_cutmixed1 = torch.cat((1-mask_u_w_cutmixed1, mask_u_w_cutmixed1), dim=1)
                conf_u_w_cutmixed1 = torch.squeeze(conf_u_w_cutmixed1, dim=1)

                # s2
                if random.uniform(0, 1) < 0.5:
                    MixMask = self._returnCutMask(pseudomask_un_t.shape[2:4], pseudomask_un_t.shape[0], cut_type="cut")
                    strong_parameters = {"Cut": MixMask}
                else:
                    strong_parameters = {"Cut": None}
                strong_parameters["flip"] = random.randint(0, 7)
                if strong_parameters["Cut"] == None:
                    strong_parameters["ColorJitter"] = 1
                else:
                    strong_parameters["ColorJitter"] = random.uniform(0, 1)

                img_u_s2, mix_un_mask_s2 = self._strongTransform(
                                                    strong_parameters,
                                                    data=img_u_w,
                                                    target=pseudomask_un_t,
                                                    cutaug=transformsgpu.mix
                                                    )
                mask_u_w_cutmixed2 = mix_un_mask_s2[:, 0:1]
                conf_u_w_cutmixed2 = mix_un_mask_s2[:, 1:2]

                mask_u_w_cutmixed2 = torch.cat((1-mask_u_w_cutmixed2, mask_u_w_cutmixed2), dim=1)
                conf_u_w_cutmixed2 = torch.squeeze(conf_u_w_cutmixed2, dim=1)

            pred_u_w_fp = self.model(img_u_w, True)

            pred_u_s1, pred_u_s2 = self.model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            # loss_u_s1 = self.criterion(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = self.criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * (conf_u_w_cutmixed1 >= self.conf_thresh)
            loss_u_s1 = torch.sum(loss_u_s1) / crit_pixel

            # loss_u_s2 = self.criterion(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = self.criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * (conf_u_w_cutmixed2 >= self.conf_thresh)
            loss_u_s2 = torch.sum(loss_u_s2) / crit_pixel

            # loss_u_w_fp = self.criterion(pred_u_w_fp, mask_u_w_mix)
            loss_u_w_fp = self.criterion_u(pred_u_w_fp, mask_u_w_mix)
            loss_u_w_fp = loss_u_w_fp * (conf_u_w_mix >= self.conf_thresh)
            loss_u_w_fp = torch.sum(loss_u_w_fp) / crit_pixel

            totalloss += (loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5)
            totalloss /= 2.0
            self.log(f"train s1 loss", loss_u_s1)
            self.log(f"train s2 loss", loss_u_s2)
            self.log(f"train fp loss", loss_u_w_fp)

        self.log(f"train sup loss", loss_x)

        self.log(f"train loss", totalloss, prog_bar=True)
        self.manual_backward(totalloss/self.accumulate_grad_batches)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt.step()
            opt.zero_grad()
        
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self._training_sch_on_step()

    def validation_step(self, batch, batch_idx):
        # return
        x, y = batch
        predmask = []
        y_pred_1 = self.model(x)
        predmask.append(torch.argmax(y_pred_1, dim=1))

        loss_1 = self.criterion(y_pred_1, y)
        self.log(f"valid loss", loss_1, prog_bar=True)

        self._evaluate(predmask, y, "valid")

    def test_step(self, batch, batch_idx):
        # return
        x, y = batch
        predmask = []
        y_pred_1 = self.model(x)
        predmask.append(torch.argmax(y_pred_1, dim=1))
        
        self._evaluate(predmask, y, "test")

    def configure_optimizers(self):
        opts = []
        schs = []

        optimizer1 = self._initoptimizer(self.model.parameters())
        scheduler1 = self._initscheduler(optimizer1)
        opts.append(optimizer1)
        schs.append(scheduler1)

        return opts, schs

    def _initmetrics(self):
        return [
            smputils.metrics.IoU(),
        ]
