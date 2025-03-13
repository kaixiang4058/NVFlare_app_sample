import torch
import torch.nn.functional as F
import copy

from .modelbase import BaseModel

class MRSupModel(BaseModel):
    def __init__(self, traincfg):
        super().__init__(traincfg)

        self.unflatten_json(self.traincfg['branch1'])
        self.branch1 = self._initmodel(self.traincfg['branch1'])

    def forward(self, x, lr, step=1):
        # in lightning, forward defines the prediction/inference actions
        """
        Args:
            x       input tensor
            step    predict branch
        """
        y_pred = self.branch1(x, lr)
        y_pred = torch.argmax(y_pred, dim=1)

        return y_pred

    def training_step(self, batch, batch_idx):
        image, mask, lrimage = batch["label"]

        opt = self.optimizers()
        # supervised
        predmask = self.branch1(image, lrimage)
        trainloss = self.criterion(predmask, mask)

        # logging
        self.log(f"train loss", trainloss, prog_bar=True)

        self._evaluate(
            predmask=torch.argmax(predmask, dim=1),
            mask=mask,
            stage="train")

        # backwarding
        totalloss = trainloss
        self.manual_backward(totalloss / self.accumulate_grad_batches)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt.step()
            opt.zero_grad()
        
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self._training_sch_on_step()

    def validation_step(self, batch, batch_idx):
        # return
        image, mask, lrimage = batch

        predmask = self.branch1(image, lrimage)
        loss = self.criterion(predmask, mask)
        self.log(f"valid loss", loss, prog_bar=True)

        self._evaluate(
            predmask=torch.argmax(predmask, dim=1),
            mask=mask,
            stage="valid")

    def test_step(self, batch, batch_idx):
        # return
        image, mask, lrimage = batch

        predmask = self.branch1(image, lrimage)

        self._evaluate(
            predmask=torch.argmax(predmask, dim=1),
            mask=mask,
            stage="test")

    @torch.no_grad()
    def _evaluate(self, predmask, mask, stage:str):
        sync_dist = False if stage == "train" else True
        for metric_fn in self.metrics:
            metric_value = metric_fn(predmask, mask)
            self.log(f'{stage} {metric_fn.__name__}', metric_value, sync_dist=sync_dist)
            if stage == "train":
                break

    def configure_optimizers(self):
        opts = []
        schs = []

        optimizer1 = self._initoptimizer(self.branch1.parameters())
        scheduler1 = self._initscheduler(optimizer1)
        opts.append(optimizer1)
        schs.append(scheduler1)

        return opts, schs
