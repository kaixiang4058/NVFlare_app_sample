import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch.utils as smputils
import math
import numpy as np
import scipy.ndimage


from .modelbase import BaseModel

class MTModel(BaseModel):
    def __init__(self, traincfg):
        super().__init__(traincfg)

        self.unflatten_json(self.traincfg['branch1'])
        self.traincfg['branch2'] = self.traincfg['branch1']

        self.s_model = self._initmodel(self.traincfg['branch1'])
        self.t_model = self._initmodel(self.traincfg['branch2'])

        # create criterions
        # TODO: support more types of the consistency criterion
        self.cons_criterion = nn.MSELoss()

    def forward(self, x, step=1):
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

        lbs = len(x)
        x = torch.concat((x, x_un), dim=0)

        predmask = []
        s_opt = self.optimizers()

        # calculate the ramp-up coefficient of the dynamic consistency constraint
        dc_rampup_epochs = 1
        cur_steps = self.trainer.global_step
        total_steps = self.steps_per_epoch * dc_rampup_epochs
        rampup_scale = self.sigmoid_rampup(cur_steps, total_steps)

        # forward the student model
        s_pred = self.s_model.forward(x)

        # calculate the supervised task constraint on the labeled data
        l_s_pred = s_pred[:lbs]

        s_task_loss = self.criterion(l_s_pred, y)
        self.log(f"train 1 sup loss", s_task_loss)
        predmask.append(torch.argmax(l_s_pred, dim=1))

        # forward the teacher model
        with torch.no_grad():
            t_pred = self.t_model.forward(x)
        
            # calculate 't_task_loss' for recording
            l_t_pred = t_pred[:lbs]
            t_task_loss = self.criterion.forward(l_t_pred, y)
            self.log(f"train 2 sup loss", t_task_loss)
            t_pseudo_gt = torch.argmax(t_pred, dim=1)
            predmask.append(t_pseudo_gt[:lbs])
        
        
        cons_loss = self.cons_criterion(s_pred[lbs:, ...], t_pseudo_gt[lbs:, ...]) * rampup_scale * self.consistencyratio

        self.log(f"train cr loss", cons_loss)

        self._evaluate(predmask, y, "train")
        # backward and update the student model
        totalloss = s_task_loss + cons_loss

        self.log(f"train loss", totalloss)
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
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)

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
