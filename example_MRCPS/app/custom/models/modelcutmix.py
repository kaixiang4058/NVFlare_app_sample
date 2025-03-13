import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import segmentation_models_pytorch.utils as smputils

from . import transformsgpu
import random 
from .modelbase import BaseModel

class CutMixModel(BaseModel):
    def __init__(self, traincfg):
        super().__init__(traincfg)

        self.unflatten_json(self.traincfg['branch1'])
        self.traincfg['branch2'] = copy.deepcopy(self.traincfg['branch1'])

        self.s_model = self._initmodel(self.traincfg['branch1'])
        self.t_model = self._initmodel(self.traincfg['branch2'])

        # create criterions
        # TODO: support more types of the consistency criterion
        self.cons_criterion = nn.MSELoss()

    def forward(self, x, step=1):
        # in lightning, forward defines the prediction/inference actions
        """
        Args:
            x       input tensor
            step    predict branch
        """
        if step==1:
            model = self.t_model
        elif step==2:
            model = self.s_model

        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim=1)

        return y_pred

    def training_step(self, batch, batch_idx):
        x, y = batch["label"]
        x_un = batch["unlabel"]

        predmask = []
        s_opt = self.optimizers()

        # calculate the ramp-up coefficient of the dynamic consistency constraint
        dc_rampup_epochs = 1
        cur_steps = self.trainer.global_step
        total_steps = self.steps_per_epoch * dc_rampup_epochs
        rampup_scale = self.sigmoid_rampup(cur_steps, total_steps)

        # supervised partial
        l_s_pred = self.s_model(x)
        s_sup_loss = self.criterion(l_s_pred, y)
        self.log(f"train sup loss", s_sup_loss)

        predmask.append(torch.argmax(l_s_pred, dim=1))
        

        # forward the teacher model
        with torch.no_grad():
            predmask.append(torch.argmax(self.t_model(x), dim=1))

            pseudomask_un_t = torch.argmax(self.t_model(x_un), dim=1)
            # calculate 't_task_loss' for recording
        
            pseudomask_un_t = torch.unsqueeze(pseudomask_un_t, dim=1)
            
            MixMask = self._returnCutMask(pseudomask_un_t.shape[2:4], pseudomask_un_t.shape[0], cut_type="cut")
            strong_parameters = {"Cut": MixMask}
            strong_parameters["flip"] = random.randint(0, 7)
            strong_parameters["ColorJitter"] = random.uniform(0, 1)

            mix_un_img, mix_un_mask = self._strongTransform(
                                                strong_parameters,
                                                data=x_un,
                                                target=pseudomask_un_t,
                                                cutaug=transformsgpu.mix
                                                )
            mix_un_mask = torch.cat((1-mix_un_mask, mix_un_mask), dim=1)
        
        mix_pred_s = self.s_model(mix_un_img)
        cr_loss_s = self.cons_criterion(mix_pred_s, mix_un_mask) * self.consistencyratio * rampup_scale
        
        self.log(f"train cr loss", cr_loss_s)

        self._evaluate(predmask, y, "train")

        # backward and update the student model
        totalloss = s_sup_loss + cr_loss_s

        self.log(f"train loss", totalloss, prog_bar=True)
        self.manual_backward(totalloss/self.accumulate_grad_batches)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            s_opt.step()
            s_opt.zero_grad()
        
        self._update_ema_variables(self.s_model, self.t_model, 0.999)

        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self._training_sch_on_step()

    def _update_ema_variables(self, s_model, t_model, ema_decay):
        # update the teacher model by exponential moving average
        ema_decay = min(1 - 1 / (self.trainer.global_step + 1), ema_decay)
        for t_param, s_param in zip(t_model.parameters(), s_model.parameters()):
            t_param.data.mul_(ema_decay).add_(s_param.data, alpha=(1 - ema_decay))

    def validation_step(self, batch, batch_idx):
        # return
        x, y = batch
        predmask = []
        y_pred_1 = self.t_model(x)
        predmask.append(torch.argmax(y_pred_1, dim=1))

        loss_1 = self.criterion(y_pred_1, y)
        self.log(f"valid loss", loss_1, prog_bar=True)

        self._evaluate(predmask, y, "valid")

    def test_step(self, batch, batch_idx):
        # return
        x, y = batch
        predmask = []
        y_pred_1 = self.t_model(x)
        predmask.append(torch.argmax(y_pred_1, dim=1))
        
        self._evaluate(predmask, y, "test")

    def configure_optimizers(self):
        opts = []
        schs = []

        optimizer1 = self._initoptimizer(self.s_model.parameters())
        scheduler1 = self._initscheduler(optimizer1)
        opts.append(optimizer1)
        schs.append(scheduler1)

        return opts, schs

    def _initmetrics(self):
        return [
            smputils.metrics.IoU(),
        ]
