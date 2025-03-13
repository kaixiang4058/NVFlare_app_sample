import torch
import copy
import segmentation_models_pytorch.utils as smputils

from .modelbase import BaseModel

class CPSModel(BaseModel):
    def __init__(self, traincfg):
        super().__init__(traincfg)

        self.unflatten_json(self.traincfg['branch1'])
        self.branch1 = self._initmodel(self.traincfg['branch1'])

        self.unflatten_json(self.traincfg['branch2'])
        self.branch2 = self._initmodel(self.traincfg['branch2'])

    def training_step(self, batch, batch_idx):
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
            y_pred_un_1 = self.branch1(x_un)
            y_pred_un_2 = self.branch2(x_un)
            pseudomask_un_1 = torch.argmax(y_pred_un_1, dim=1)
            pseudomask_un_2 = torch.argmax(y_pred_un_2, dim=1)
            cps_loss_1 = self.criterion(y_pred_un_1, pseudomask_un_2) * self.consistencyratio
            cps_loss_2 = self.criterion(y_pred_un_2, pseudomask_un_1) * self.consistencyratio
            
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

    def validation_step(self, batch, batch_idx):
        # return
        x, y = batch
        predmask = []
        y_pred_1 = self.branch1(x)
        predmask.append(torch.argmax(y_pred_1, dim=1))
        y_pred_2 = self.branch2(x)
        predmask.append(torch.argmax(y_pred_2, dim=1))

        loss_1 = self.criterion(y_pred_1, y)
        self.log(f"valid 1 loss", loss_1)
        loss_2 = self.criterion(y_pred_2, y)
        self.log(f"valid 2 loss", loss_2)

        self.log(f"valid loss", (loss_1 + loss_2) / 2, prog_bar=True)

        self._evaluate(predmask, y, "valid")

        predensem = []
        voting = y_pred_1.softmax(1) + y_pred_2.softmax(1)
        predensem.append(torch.argmax(voting, dim=1))

        self._evaluate(predensem, y, "valid ens")

    def test_step(self, batch, batch_idx):
        # return
        x, y = batch
        predmask = []
        y_pred_1 = self.branch1(x)
        predmask.append(torch.argmax(y_pred_1, dim=1))
        y_pred_2 = self.branch2(x)
        predmask.append(torch.argmax(y_pred_2, dim=1))
        
        results = self._evaluate(predmask, y, "test")

        predensem = []
        voting = y_pred_1.softmax(1) + y_pred_2.softmax(1)
        predensem.append(torch.argmax(voting, dim=1))

        self._evaluate(predensem, y, "test ens")

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

        return opts, schs

    def _initmetrics(self):
        return [
            smputils.metrics.IoU(),
            smputils.metrics.Recall(),
        ]