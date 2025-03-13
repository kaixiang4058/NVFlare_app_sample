import torch
import torch.nn.functional as F

import segmentation_models_pytorch.utils as smputils
import numpy as np

from .modelbase import BaseModel

from models.u2pl.dataset.augmentation import generate_unsup_data
from models.u2pl.utils.loss_helper import (
    compute_contra_memobank_loss,
    compute_unsupervised_loss,
)
import yaml
from models.u2pl.models.segformer import SegFormer
from models.u2pl.models.unet import Unet

from models.u2pl.models.model_helper import ModelBuilder
from models.u2pl.utils.utils import label_onehot

class U2PLModel(BaseModel):
    def __init__(self, traincfg):
        super().__init__(traincfg)


        cfg = yaml.load(open("models/u2pl/config.yaml", "r"), Loader=yaml.Loader)
        self.cfg = cfg

        self.unflatten_json(self.traincfg['branch1'])

        model_type = self.traincfg['branch1']['model'].pop('type')
        torch.manual_seed(self.traincfg['branch1']['model_seed'])

        feat_chan = 256
        if model_type == "SegFormer":
            self.model = SegFormer(**self.traincfg['branch1']['model'])
            self.model_teacher = SegFormer(**self.traincfg['branch1']['model'])

        elif model_type == "Unet":
            self.model = Unet(**self.traincfg['branch1']['model'])
            if cfg["net"].get("sync_bn", True):
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model_teacher = Unet(**self.traincfg['branch1']['model'])
            feat_chan = 64

        elif model_type == "DeepLabV3Plus":
            self.model = ModelBuilder(cfg["net"])

            if cfg["net"].get("sync_bn", True):
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

            # Teacher model
            self.model_teacher = ModelBuilder(cfg["net"])

        for p in self.model_teacher.parameters():
            p.requires_grad = False

        self.ema_decay_origin = 0.99

        # build class-wise memory bank
        self.memobank = []
        self.queue_ptrlis = []
        self.queue_size = []
        for i in range(cfg["net"]["num_classes"]):
            self.memobank.append([torch.zeros(0, feat_chan)])
            self.queue_size.append(30000)
            self.queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
        self.queue_size[0] = 50000

        # build prototype
        self.prototype = torch.zeros(
            (
                cfg["net"]["num_classes"],
                cfg["trainer"]["contrastive"]["num_queries"],
                1,
                feat_chan,
            )
        )

    def forward(self, x, step=1):
        """
        Args:
            x       input tensor
            step    predict branch
        """
        if step==1:
            model = self.model_teacher
        elif step==2:
            model = self.model

        y_pred = model(x)["pred"]
        y_pred = F.interpolate(
                y_pred, x.shape[-2:], mode="bilinear", align_corners=True
        )
        y_pred = torch.argmax(y_pred, dim=1)

        return y_pred

    def training_step(self, batch, batch_idx):
        image_l, label_l = batch["label"]
        image_u = batch["unlabel"]

        _, h, w = label_l.size()

        predmask = []
        s_opt = self.optimizers()

        if self.current_epoch < self.cfg["trainer"].get("sup_only_epoch", 1):
            contra_flag = "none"
            # forward
            outs = self.model(image_l)
            pred, rep = outs["pred"], outs["rep"]
            pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

            # supervised loss
            sup_loss = self.criterion(pred, label_l)
            predmask.append(torch.argmax(pred, dim=1))

            self.model_teacher.train()
            _ = self.model_teacher(image_l)

            unsup_loss = 0 * rep.sum()
            contra_loss = 0 * rep.sum()
        else:
            if self.current_epoch == self.cfg["trainer"].get("sup_only_epoch", 1):
                # copy student parameters to teacher
                with torch.no_grad():
                    for t_params, s_params in zip(
                        self.model_teacher.parameters(), self.model.parameters()
                    ):
                        t_params.data = s_params.data

            # generate pseudo labels first
            self.model_teacher.eval()
            pred_u_teacher = self.model_teacher(image_u)["pred"]
            pred_u_teacher = F.interpolate(
                pred_u_teacher, (h, w), mode="bilinear", align_corners=True
            )
            pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
            logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)

            # apply strong data augmentation: cutout, cutmix, or classmix
            if np.random.uniform(0, 1) < 0.5 and self.cfg["trainer"]["unsupervised"].get(
                "apply_aug", False
            ):
                image_u_aug, label_u_aug, logits_u_aug = generate_unsup_data(
                    image_u,
                    label_u_aug.clone(),
                    logits_u_aug.clone(),
                    mode=self.cfg["trainer"]["unsupervised"]["apply_aug"],
                )
            else:
                image_u_aug = image_u

            # forward
            num_labeled = len(image_l)
            image_all = torch.cat((image_l, image_u_aug))
            outs = self.model(image_all)
            pred_all, rep_all = outs["pred"], outs["rep"]
            pred_l, pred_u = pred_all[:num_labeled], pred_all[num_labeled:]
            pred_l_large = F.interpolate(
                pred_l, size=(h, w), mode="bilinear", align_corners=True
            )
            pred_u_large = F.interpolate(
                pred_u, size=(h, w), mode="bilinear", align_corners=True
            )

            # supervised loss
            sup_loss = self.criterion(pred_l_large, label_l.clone())
            predmask.append(torch.argmax(pred_l_large, dim=1))

            # teacher forward
            self.model_teacher.train()
            with torch.no_grad():
                out_t = self.model_teacher(image_all)
                pred_all_teacher, rep_all_teacher = out_t["pred"], out_t["rep"]
                prob_all_teacher = F.softmax(pred_all_teacher, dim=1)
                prob_l_teacher, prob_u_teacher = (
                    prob_all_teacher[:num_labeled],
                    prob_all_teacher[num_labeled:],
                )

                pred_u_teacher = pred_all_teacher[num_labeled:]
                pred_u_large_teacher = F.interpolate(
                    pred_u_teacher, size=(h, w), mode="bilinear", align_corners=True
                )

            # unsupervised loss
            drop_percent = self.cfg["trainer"]["unsupervised"].get("drop_percent", 100)
            percent_unreliable = (100 - drop_percent) * (1 - self.current_epoch / self.cfg["trainer"]["epochs"])
            drop_percent = 100 - percent_unreliable
            unsup_loss = (
                    compute_unsupervised_loss(
                        pred_u_large,
                        label_u_aug.clone(),
                        drop_percent,
                        pred_u_large_teacher.detach(),
                    )
                    * self.cfg["trainer"]["unsupervised"].get("loss_weight", 1)
            )

            # contrastive loss using unreliable pseudo labels
            contra_flag = "none"
            if self.cfg["trainer"].get("contrastive", False):
                cfg_contra = self.cfg["trainer"]["contrastive"]
                contra_flag = "{}:{}".format(
                    cfg_contra["low_rank"], cfg_contra["high_rank"]
                )
                alpha_t = cfg_contra["low_entropy_threshold"] * (
                    1 - self.current_epoch / self.cfg["trainer"]["epochs"]
                )

                with torch.no_grad():
                    prob = torch.softmax(pred_u_large_teacher, dim=1)
                    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

                    low_thresh = np.percentile(
                        entropy[label_u_aug != 255].cpu().numpy().flatten(), alpha_t
                    )
                    low_entropy_mask = (
                        entropy.le(low_thresh).float() * (label_u_aug != 255).bool()
                    )

                    high_thresh = np.percentile(
                        entropy[label_u_aug != 255].cpu().numpy().flatten(),
                        100 - alpha_t,
                    )
                    high_entropy_mask = (
                        entropy.ge(high_thresh).float() * (label_u_aug != 255).bool()
                    )

                    low_mask_all = torch.cat(
                        (
                            (label_l.unsqueeze(1) != 255).float(),
                            low_entropy_mask.unsqueeze(1),
                        )
                    )

                    low_mask_all = F.interpolate(
                        low_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )
                    # down sample

                    if cfg_contra.get("negative_high_entropy", True):
                        contra_flag += " high"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                high_entropy_mask.unsqueeze(1),
                            )
                        )
                    else:
                        contra_flag += " low"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                torch.ones(logits_u_aug.shape)
                                .float()
                                .unsqueeze(1)
                                .cuda(),
                            ),
                        )
                    high_mask_all = F.interpolate(
                        high_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )  # down sample

                    # down sample and concat
                    label_l_small = F.interpolate(
                        label_onehot(label_l, self.cfg["net"]["num_classes"]),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    )
                    label_u_small = F.interpolate(
                        label_onehot(label_u_aug, self.cfg["net"]["num_classes"]),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    )

                if not cfg_contra.get("anchor_ema", False):
                    new_keys, contra_loss = compute_contra_memobank_loss(
                        rep_all,
                        label_l_small.long(),
                        label_u_small.long(),
                        prob_l_teacher.detach(),
                        prob_u_teacher.detach(),
                        low_mask_all,
                        high_mask_all,
                        cfg_contra,
                        self.memobank,
                        self.queue_ptrlis,
                        self.queue_size,
                        rep_all_teacher.detach(),
                    )
                else:
                    self.prototype, new_keys, contra_loss = compute_contra_memobank_loss(
                        rep_all,
                        label_l_small.long(),
                        label_u_small.long(),
                        prob_l_teacher.detach(),
                        prob_u_teacher.detach(),
                        low_mask_all,
                        high_mask_all,
                        cfg_contra,
                        self.memobank,
                        self.queue_ptrlis,
                        self.queue_size,
                        rep_all_teacher.detach(),
                        self.prototype,
                    )

            else:
                raise ValueError("Not using Contra Loss")
                contra_loss = 0 * rep_all.sum()

        loss = sup_loss + unsup_loss + contra_loss
        self.log(f"train sup loss", sup_loss)
        self.log(f"train unsup loss", unsup_loss)
        self.log(f"train contra loss", contra_loss)
        self.log(f"train loss", loss, prog_bar=True)
        self._evaluate(predmask, label_l, "train")
        
        self.manual_backward(loss/self.accumulate_grad_batches)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            s_opt.step()
            s_opt.zero_grad()
        
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self._training_sch_on_step()

        # update teacher model with EMA
        if self.current_epoch >= self.cfg["trainer"].get("sup_only_epoch", 1):
            with torch.no_grad():
                ema_decay = min(
                    1
                    - 1
                    / (
                        self.global_step
                        - self.steps_per_epoch * self.cfg["trainer"].get("sup_only_epoch", 1)
                        + 1
                    ),
                    self.ema_decay_origin,
                )
                for t_params, s_params in zip(
                    self.model_teacher.parameters(), self.model.parameters()
                ):
                    t_params.data = (
                        ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                    )

    def validation_step(self, batch, batch_idx):
        # return
        x, y = batch
        predmask = []

        if self.current_epoch < self.cfg["trainer"].get("sup_only_epoch", 1):
            model = self.model
        else:
            model = self.model_teacher

        outs = model(x)
        y_pred_1 = outs["pred"]
        y_pred_1 = F.interpolate(
            y_pred_1, y.shape[1:], mode="bilinear", align_corners=True
        )
        predmask.append(torch.argmax(y_pred_1, dim=1))

        loss_1 = self.criterion(y_pred_1, y)
        self.log(f"valid loss", loss_1, prog_bar=True)

        self._evaluate(predmask, y, "valid")

    def test_step(self, batch, batch_idx):
        # return
        x, y = batch
        predmask = []

        model = self.model_teacher

        outs = model(x)
        y_pred_1 = outs["pred"]
        y_pred_1 = F.interpolate(
            y_pred_1, y.shape[1:], mode="bilinear", align_corners=True
        )
        predmask.append(torch.argmax(y_pred_1, dim=1))

        loss_1 = self.criterion(y_pred_1, y)
        self.log(f"test loss", loss_1, prog_bar=True)

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
