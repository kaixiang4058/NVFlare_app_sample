import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.optim import lr_scheduler

class UnetModel(pl.LightningModule):
    def __init__(self, num_class=2, lr=None, batch_size=None, **kwargs):
        super().__init__()
        self.net1 = UNet(random_seed=52)
        self.net2 = UNet(random_seed=12)
        self.crit = nn.CrossEntropyLoss()
        self.lr = lr
        self.batch_size = batch_size
        self.automatic_optimization = False

    def forward(self, x):
        seg1, logits1, cam1 = self.net1(x)
        seg2, logits2, cam2 = self.net2(x)
        return seg1[:self.batch_size], seg1[self.batch_size:], seg2[:self.batch_size], seg2[self.batch_size:], \
            logits1[:self.batch_size], logits1[self.batch_size:], logits2[:self.batch_size], logits2[self.batch_size:], \
            cam1[:self.batch_size], cam1[self.batch_size:], cam2[:self.batch_size], cam2[self.batch_size:]

    def forward(self, x, step=1):
        y_pred = getattr(self, f'net{step}')(x)[0]
        y_pred = torch.argmax(y_pred, dim=1)    

        return y_pred

    def dice_coeff(self, pred, target):
        smooth = 1e-15
        target = target[:, 1, :, :]
        pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
        num = pred.size(0)
        m1 = pred.view(num, -1)  # Flatten
        m2 = target.view(num, -1)  # Flatten
        intersection = (m1 * m2).sum()
        return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

    def binary_mean_iou(self, logits, targets):
        EPSILON = 1e-15
        targets = targets[:,1 ,:, :]
        output = torch.argmax(logits, dim=1)
        if output.shape != targets.shape:
            targets = torch.squeeze(targets, 1)
        intersection = torch.logical_and(output, targets)
        union = torch.logical_or(output, targets)
        result = (torch.sum(intersection) + EPSILON) / (torch.sum(union) +EPSILON)
        return result

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        consistency = 1.0
        consistency_rampup = 0

        def sigmoid_rampup(current, rampup_length):
            if rampup_length == 0:
                return consistency
            else:
                current = np.clip(current, 0.0, rampup_length)
                phase = 1.0 - current / rampup_length
                return float(np.exp(-5.0 * phase * phase))
        return consistency * sigmoid_rampup(epoch, consistency_rampup)

    def training_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()
        opt1.zero_grad()
        opt2.zero_grad()

        img, mask, label = batch['fully']
        img_w = batch['weakly']
#         consistency_weight = 1 if self.current_epoch < 28 else 10
        consistency_weight = self.get_current_consistency_weight(self.current_epoch)
        imgs = torch.cat((img, img_w))
        # net1_logits = 0, net1_logits_w = 1, net2_logits = 2, net2_logits_w = 3
        # class1 = 4, class1_w = 5, class2 = 6, class2_w = 7
        # cam1 = 8, cam1_w = 9, cam2 = 10, cam2_w = 11
        output = self(imgs)

        with torch.no_grad():
            net1_psu = torch.argmax(output[0].detach().softmax(1), dim=1)
            net1_psu_w = torch.argmax(output[1].detach().softmax(1), dim=1)
            net2_psu = torch.argmax(output[2].detach().softmax(1), dim=1)
            net2_psu_w = torch.argmax(output[3].detach().softmax(1), dim=1)
            class1_psu = torch.argmax(output[4].detach().softmax(1), dim=1)
            class1_psu_w = torch.argmax(output[5].detach().softmax(1), dim=1)
            class2_psu = torch.argmax(output[6].detach().softmax(1), dim=1)
            class2_psu_w = torch.argmax(output[7].detach().softmax(1), dim=1)
#         ---segmentation---
        seg_loss1 = self.crit(output[0], mask)  # supervised
        seg_cu_loss1 = consistency_weight * self.crit(output[1], net2_psu_w)  # cross unlabel
        seg_cl_loss1 = self.crit(output[0], net2_psu)  # cross label

#         ---classification---
        cla_loss1 = self.crit(output[4], label)  # supervised
        cla_cu_loss1 = self.crit(output[5], class2_psu_w)
        cla_cl_loss1 = self.crit(output[4], class2_psu)
#         ---task level---
        task_loss1 = self.crit(output[8], net1_psu) + self.crit(output[9], net1_psu_w)
        loss1 = seg_loss1 + seg_cu_loss1 + seg_cl_loss1 + cla_loss1 + cla_cu_loss1 + cla_cl_loss1 + task_loss1

#         ---segmentation---
        seg_loss2 = self.crit(output[2], mask)
        seg_cu_loss2 = consistency_weight * self.crit(output[3], net1_psu_w)
        seg_cl_loss2 = self.crit(output[2], net1_psu)
#         ---classification---
        cla_loss2 = self.crit(output[6], label)  # supervised
        cla_cu_loss2 = self.crit(output[7], class1_psu_w)
        cla_cl_loss2 = self.crit(output[6], class1_psu)
#         ---task level---
        task_loss2 = self.crit(output[10], net2_psu) + self.crit(output[11], net2_psu_w)
        loss2 = seg_loss2 + seg_cu_loss2 + seg_cl_loss2 + cla_loss2 + cla_cu_loss2 + cla_cl_loss2 + task_loss2

        with torch.no_grad():
            mask = mask.type(torch.cuda.IntTensor)
            net1_iou = self.binary_mean_iou(
                torch.softmax(output[0].detach(), dim=1), mask)
            net2_iou = self.binary_mean_iou(
                torch.softmax(output[2].detach(), dim=1), mask)
            class1_acc = accuracy(output[4].detach().softmax(1), label)
            class2_acc = accuracy(output[6].detach().softmax(1), label)
            dice1 = self.dice_coeff(output[0].detach(), mask)  # supervised
            dice2 = self.dice_coeff(output[2].detach(), mask)
        loss = loss1 + loss2
        self.manual_backward(loss)
#         self.manual_backward(loss1)
        opt1.step()
#         self.manual_backward(loss2)
        opt2.step()
            
        self.log('train_net1_loss', loss1, prog_bar=True, on_epoch=True)
        self.log('train_net1_iou', net1_iou, prog_bar=True, on_epoch=True)
        self.log('train_net2_loss', loss2, prog_bar=True, on_epoch=True)
        self.log('train_net2_iou', net2_iou, prog_bar=True, on_epoch=True)
        self.log('train_net1_acc', class1_acc, on_epoch=True)
        self.log('train_net2_acc', class2_acc, on_epoch=True)
        self.log('task_loss1', task_loss1, on_step=False, on_epoch=True)
        self.log('task_loss2', task_loss2, on_step=False, on_epoch=True)
        self.log('dice1', dice1)
        self.log('dice2', dice2)
        # -------------
        self.log('seg_loss1', seg_loss1, on_step=False, on_epoch=True)
        self.log('seg_cu_loss1', seg_cu_loss1, on_step=False, on_epoch=True)
        self.log('seg_cl_loss1', seg_cl_loss1, on_step=False, on_epoch=True)
        self.log('cla_loss1', cla_loss1, on_step=False, on_epoch=True)
        self.log('cla_cu_loss1', cla_cu_loss1, on_step=False, on_epoch=True)
        self.log('cla_cl_loss1', cla_cl_loss1, on_step=False, on_epoch=True)
        self.log('seg_loss2', seg_loss2, on_step=False, on_epoch=True)
        self.log('seg_cu_loss2', seg_cu_loss2, on_step=False, on_epoch=True)
        self.log('seg_cl_loss2', seg_cl_loss2, on_step=False, on_epoch=True)
        self.log('cla_loss2', cla_loss2, on_step=False, on_epoch=True)
        self.log('cla_cu_loss2', cla_cu_loss2, on_step=False, on_epoch=True)
        self.log('cla_cl_loss2', cla_cl_loss2, on_step=False, on_epoch=True)

    def training_epoch_end(self, outputs):
        def check(sch, monitor):
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step(self.trainer.callback_metrics[monitor])
            else:
                sch.step()
        sch1, sch2 = self.lr_schedulers()
        check(sch1, "train_net1_loss")
        check(sch2, "train_net2_loss")

    def validation_step(self, batch, batch_idx):
        img, mask, label = batch
        # net1_logits = 0, net1_logits_w = 1, net2_logits = 2, net2_logits_w = 3
        # class1 = 4, class1_w = 5, class2 = 6, class2_w = 7
        # cam1 = 8, cam1_w = 9, cam2 = 10, cam2_w = 11
        output = self(img)
        val_seg_loss1 = self.crit(output[0], mask)
        val_cla_loss1 = self.crit(output[4], label)
        loss1 = val_seg_loss1
        val_seg_loss2 = self.crit(output[2], mask)
        val_cla_loss2 = self.crit(output[6], label)
        loss2 = val_seg_loss2

        loss = loss1 + loss2
        with torch.no_grad():
            mask = mask.type(torch.cuda.IntTensor)
            net1_iou = self.binary_mean_iou(
                torch.softmax(output[0].detach(), dim=1), mask)
            net2_iou = self.binary_mean_iou(
                torch.softmax(output[2].detach(), dim=1), mask)
            class1_acc = accuracy(output[4].detach().softmax(1), label)
            class2_acc = accuracy(output[6].detach().softmax(1), label)

        self.log('valid_net1_iou', net1_iou, on_epoch=True)
        self.log('valid_net2_iou', net2_iou, on_epoch=True)
        self.log('valid_loss_epoch', loss, on_epoch=True)
        self.log('valid_loss1_epoch', loss1, on_epoch=True)
        self.log('valid_loss2_epoch', loss2, on_epoch=True)
        self.log('valid_net1_acc', class1_acc, on_epoch=True)
        self.log('valid_net2_acc', class2_acc, on_epoch=True)
        # -----------------------
        self.log('val_seg_loss1', val_seg_loss1, on_step=False, on_epoch=True)
        self.log('val_cla_loss1', val_cla_loss1, on_step=False, on_epoch=True)
        self.log('val_seg_loss2', val_seg_loss2, on_step=False, on_epoch=True)
        self.log('val_cla_loss2', val_cla_loss2, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        img, mask, label = batch
        output = self(img)
        with torch.no_grad():
            mask = mask.type(torch.cuda.IntTensor)
            net1_iou = self.binary_mean_iou(
                torch.softmax(output[0].detach(), dim=1), mask)
            net2_iou = self.binary_mean_iou(
                torch.softmax(output[2].detach(), dim=1), mask)
        self.log('test_net1_iou', net1_iou, on_epoch=True)
        self.log('test_net2_iou', net2_iou, on_epoch=True)

    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(self.net1.parameters(), self.lr, weight_decay=1e-4)
        scheduler1 = lr_scheduler.CosineAnnealingWarmRestarts(optimizer1, T_0=4, T_mult=2)
        optimizer2 = torch.optim.Adam(self.net2.parameters(), self.lr, weight_decay=1e-4)
        scheduler2 = lr_scheduler.CosineAnnealingWarmRestarts(optimizer2, T_0=4, T_mult=2)
        return ({
            "optimizer": optimizer1,
            "lr_scheduler": {
                "scheduler": scheduler1,
                #                                         "monitor": "train_net1_loss",
            }
        },
            {
            "optimizer": optimizer2,
            "lr_scheduler": {
                "scheduler": scheduler2,
                #                                         "monitor": "train_net2_loss",
            }
        },
        )

class UNet(nn.Module):
    def __init__(self, num_class=2, random_seed=None):
        super().__init__()
        if random_seed is not None:
            self.seed_everything(random_seed)
        self.net = smp.Unet(
            encoder_name="efficientnet-b3",
            in_channels=3,
            classes=num_class,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(384, num_class, 1, bias=False),
            nn.BatchNorm2d(num_class),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.net.apply(self.initialize)
        self.conv.apply(self.initialize)
        
    def forward(self, x):
        x = self.net.encoder(x)
        cam = self.conv(x[-1])
        x = self.net.decoder(*x)
        seg = self.net.segmentation_head(x)        
        logits = self.avgpool(cam)
        logits = logits.view(logits.size(0), -1)
        
        b, c, h, w = seg.size()        
        cam = F.interpolate(cam, (h,w), mode='bilinear', align_corners=True)
        cam = self.make_cam(cam)
        return seg, logits, cam

    def make_cam(self, x, epsilon=1e-5):
        b, c, h, w = x.size()
        flat_x = x.view(b, c, (h * w))
        max_value = flat_x.max(axis=-1)[0].view((b, c, 1, 1))
        return F.relu(x - epsilon) / (max_value + epsilon)    
            
    def seed_everything(self, seed: int):
        import random
        import os
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def initialize(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()