import torch
import random

from .modelbase import BaseModel
import segmentation_models_pytorch.utils as smputils


class MRTRIMixModel(BaseModel):
    def __init__(self, traincfg):
        super().__init__(traincfg)
        modelcfg = {
            'model_seed' : traincfg["expset"]["env_seed"],
            'model' : {
                'encoder_name_1' : 'resnest26d',
                'encoder_name_2' : 'nvidia/mit-b1',
                'lrbackbone' : "nvidia/mit-b1",
                'lrscale' : traincfg['expset']['lrratio'],
                'in_chans' : 3,
                'classes' : 2,
            }
        }
        torch.manual_seed(modelcfg['model_seed'])

        self.net = MRUTriNet(**modelcfg['model'])

    def forward(self, x, lrx, step=1):
        if step==0:
            pred1, pred2 = self.net(x, lrx)
            return torch.argmax(
                pred1.softmax(1) + pred2.softmax(1)
                , dim=1)
        elif step==3:
            pred1, pred2 = self.net(x, lrx)
            pred1 = pred1.softmax(1)
            pred2 = pred2.softmax(1)
            return torch.argmax(pred1+pred2, dim=1), torch.argmax(pred1, dim=1), torch.argmax(pred2, dim=1)
        else:
            return torch.argmax(self.net(x, lrx), dim=1)

    def training_step(self, batch, batch_idx):
        image, mask, lrimage = batch["label"]

        predmask = []
        opt1 = self.optimizers()

        # supervised
        y_pred_1_sup, y_pred_2_sup = self.net(image, lrimage)
        
        sup_loss_1 = self.criterion(y_pred_1_sup, mask)
        sup_loss_2 = self.criterion(y_pred_2_sup, mask)
        self.log(f"train 1 sup loss", sup_loss_1)
        self.log(f"train 2 sup loss", sup_loss_2)
        totalloss = sup_loss_1 + sup_loss_2

        predmask.append(torch.argmax(y_pred_1_sup, dim=1))
        predmask.append(torch.argmax(y_pred_2_sup, dim=1))

        if "unlabel" in batch:
            image_un, lrimage_un = batch["unlabel"]

            with torch.no_grad():
                y_pred_un_1, y_pred_un_2 = self.net(image_un, lrimage_un)
                pseudomask_un_1 = torch.argmax(y_pred_un_1, dim=1)
                pseudomask_un_2 = torch.argmax(y_pred_un_2, dim=1)

                pseudomask_cat = torch.cat(\
                    (torch.unsqueeze(pseudomask_un_1, dim=1), torch.unsqueeze(pseudomask_un_2, dim=1)), dim=1)
                strong_parameters = {}
                strong_parameters["flip"] = random.randint(0, 7)
                strong_parameters["ColorJitter"] = random.uniform(0, 1)

                mix_un_img, mix_un_lrimg, mix_un_mask = self._strongTransform(
                                                    strong_parameters,
                                                    data=image_un,
                                                    lrdata=lrimage_un,
                                                    target=pseudomask_cat
                                                    )
                mix_un_mask_1 = torch.squeeze(mix_un_mask[:, 0:1], dim=1).long()
                mix_un_mask_2 = torch.squeeze(mix_un_mask[:, 1:2], dim=1).long()
            mix_pred_1, mix_pred_2 = self.net(mix_un_img, mix_un_lrimg)
            cps_loss_1 = self.criterion(mix_pred_1, mix_un_mask_2) * self.consistencyratio
            cps_loss_2 = self.criterion(mix_pred_2, mix_un_mask_1) * self.consistencyratio
            
            self.log(f"train 1 cps loss", cps_loss_1)
            self.log(f"train 2 cps loss", cps_loss_2)
            totalloss += (cps_loss_1 + cps_loss_2)

        self.log(f"train loss", totalloss.item() / 2, prog_bar=True)
        
        self._evaluate(predmask, mask, "train")

        # backwarding
        totalloss /= self.accumulate_grad_batches
        self.manual_backward(totalloss)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt1.step()
            opt1.zero_grad()
        
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self._training_sch_on_step()

    def validation_step(self, batch, batch_idx):
        # return
        image, mask, lrimage = batch
        predmask = []
        y_pred_1, y_pred_2 = self.net(image, lrimage)
        predmask.append(torch.argmax(y_pred_1, dim=1))
        predmask.append(torch.argmax(y_pred_2, dim=1))

        loss_1 = self.criterion(y_pred_1, mask)
        self.log(f"valid 1 loss", loss_1)
        loss_2 = self.criterion(y_pred_2, mask)
        self.log(f"valid 2 loss", loss_2)

        self.log(f"valid loss", (loss_1 + loss_2) / 2, prog_bar=True)

        self._evaluate(predmask, mask, "valid")

        predensem = []
        voting = y_pred_1.softmax(1) + y_pred_2.softmax(1)
        predensem.append(torch.argmax(voting, dim=1))

        self._evaluate(predensem, mask, "valid ens")

    def test_step(self, batch, batch_idx):
        # return
        image, mask, lrimage = batch
        predmask = []
        y_pred_1, y_pred_2 = self.net(image, lrimage)
        predmask.append(torch.argmax(y_pred_1, dim=1))
        predmask.append(torch.argmax(y_pred_2, dim=1))
        
        self._evaluate(predmask, mask, "test")

        predensem = []
        voting = y_pred_1.softmax(1) + y_pred_2.softmax(1)
        predensem.append(torch.argmax(voting, dim=1))

        self._evaluate(predensem, mask, "test ens")

    def configure_optimizers(self):
        opts = []
        schs = []

        optimizer1 = self._initoptimizer(self.net.parameters())
        scheduler1 = self._initscheduler(optimizer1)
        opts.append(optimizer1)
        schs.append(scheduler1)

        return opts, schs

    def _initmetrics(self):
        return [
            smputils.metrics.IoU(),
        ]


import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from ..networks.module import Conv2dBnAct, MScenterMLP
from ..networks.initialize import initialize_decoder
from ..networks.decoders import UnetDecoder

from transformers import SegformerModel, SegformerConfig

class MRUTriNet(nn.Module):
    def __init__(
            self,
            encoder_name_1='resnest26d',
            encoder_name_2='nvidia/mit-b1',
            lrbackbone="nvidia/mit-b1",
            lrscale=8,
            backbone_kwargs=None,
            backbone_indices=None,
            decoder_channels=(256, 128, 64, 32, 16),
            in_chans=3,
            classes=2,
            norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        # NOTE some models need different backbone indices specified based on the alignment of features
        # and some models won't have a full enough range of feature strides to work properly.
        encoder1 = create_model(
            encoder_name_1, features_only=True, out_indices=backbone_indices, in_chans=in_chans,
            pretrained=True, **backbone_kwargs)
        # reverse channels [2048, 1024, 512, 256, 64]
        encoder1_channels = encoder1.feature_info.channels()[::-1]
        self.encoder1 = encoder1

        encoder2 = SegformerModel.from_pretrained(
            encoder_name_2,config=SegformerConfig.from_pretrained(
                encoder_name_2, output_hidden_states=True
            ))
        # reverse channels [2048, 1024, 512, 256, 64]
        encoder2_channels = encoder2.config.hidden_sizes[::-1]
        self.encoder2 = encoder2

        self.lrencoder = SegformerModel.from_pretrained(
            lrbackbone,config=SegformerConfig.from_pretrained(
                lrbackbone, output_hidden_states=True
            ))
        lr_channels = self.lrencoder.config.hidden_sizes
        # reverse channels [512, 320, 128, 64]

        self.mscenter_mlp = MScenterMLP(lr_channels, 256, lrscale)
        initialize_decoder(self.mscenter_mlp)

        self.fusionblock1 = Conv2dBnAct(
            256 * 4 + encoder1_channels[0], encoder1_channels[0] // 2, kernel_size=(1, 1))
        encoder1_channels[0] //= 2
        initialize_decoder(self.fusionblock1)

        self.fusionblock2 = Conv2dBnAct(
            256 * 4 + encoder2_channels[0], encoder2_channels[0], kernel_size=(1, 1))
        initialize_decoder(self.fusionblock2)

        self.lrscale = lrscale

        self.decoder1 = UnetDecoder(
            encoder_channels=encoder1_channels,
            decoder_channels=decoder_channels[:len(encoder1_channels)],
            final_channels=classes,
            norm_layer=norm_layer,
        )

        self.decoder2 = UnetDecoder(
            encoder_channels=encoder2_channels,
            decoder_channels=decoder_channels[:len(encoder2_channels)],
            final_channels=classes,
            norm_layer=norm_layer,
        )

    def forward(self, x, lr, step="train"):
        if step=="train" or step==0 or step==3:
            _, _, h, w = x.shape
            x1 = self.encoder1(x)
            x1.reverse()
            x2 = list(self.encoder2(x).hidden_states)
            x2.reverse()

            lr = list(self.lrencoder(lr).hidden_states)
            centerlr = self.mscenter_mlp(lr)
            
            x1[0] = self.fusionblock1(torch.cat((x1[0], centerlr), dim=1))
            x2[0] = self.fusionblock2(torch.cat((x2[0], centerlr), dim=1))

            predmask1 = self.decoder1(x1)
            predmask2 = self.decoder2(x2)

            predmask1 = F.interpolate(predmask1, size=(h, w), mode="bilinear", align_corners=False)
            predmask2 = F.interpolate(predmask2, size=(h, w), mode="bilinear", align_corners=False)

            return predmask1, predmask2

        else:
            _, _, h, w = x.shape
            x = getattr(self, f'encoder{step}')(x)
            x.reverse()

            lr = list(self.lrencoder(lr).hidden_states)
            centerlr = self.mscenter_mlp(lr)
            
            x[0] = getattr(self, f'fusionblock{step}')(torch.cat((x[0], centerlr), dim=1))

            predmask = getattr(self, f'decoder{step}')(x)

            predmask = F.interpolate(predmask, size=(h, w), mode="bilinear", align_corners=False)

            return predmask

        