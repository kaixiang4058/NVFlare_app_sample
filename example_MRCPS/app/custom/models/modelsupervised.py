import torch
import copy

from .modelbase import BaseModel

class SupModel(BaseModel):
    def __init__(self, traincfg):
        super().__init__(traincfg)

        self.unflatten_json(self.traincfg['branch1'])
        self.branch1 = self._initmodel(self.traincfg['branch1'])

    def training_step(self, batch, batch_idx):
        x, y = batch["label"]
        predmask = []
        opt = self.optimizers()
        # supervised
        y_pred = self.branch1(x)
        totalloss = self.criterion(y_pred, y)
        predmask.append(torch.argmax(y_pred, dim=1))

        # logging
        self.log(f"train loss", totalloss, prog_bar=True)
        self._evaluate(predmask, y, "train")

        # backwarding
        self.manual_backward(totalloss / self.accumulate_grad_batches)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt.step()
            opt.zero_grad()
        
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self._training_sch_on_step()

    def validation_step(self, batch, batch_idx):
        # return
        x, y = batch
        predmask = []
        y_pred_1 = self.branch1(x)
        predmask.append(torch.argmax(y_pred_1, dim=1))
        loss = self.criterion(y_pred_1, y)
        self.log(f"valid loss", loss, prog_bar=True)

        self._evaluate(predmask, y, "valid")

    def test_step(self, batch, batch_idx):
        # return
        x, y = batch
        predmask = []
        y_pred_1 = self.branch1(x)
        predmask.append(torch.argmax(y_pred_1, dim=1))
        
        self._evaluate(predmask, y, "test")

    def configure_optimizers(self):
        opts = []
        schs = []

        optimizer1 = self._initoptimizer(self.branch1.parameters())
        scheduler1 = self._initscheduler(optimizer1)
        opts.append(optimizer1)
        schs.append(scheduler1)

        return opts, schs
