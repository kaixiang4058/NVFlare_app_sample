import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model
from .module import Conv2dBnAct
from .initialize import initialize_decoder, initialize_head
from .decoders import UnetDecoder

from transformers import SegformerModel, SegformerConfig

class MRUnet(nn.Module):
    def __init__(
            self,
            encoder_name='resnest26d',
            lrbackbone='resnest26d',
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
        if "res" in encoder_name:
            encoder = create_model(
                encoder_name, features_only=True, out_indices=backbone_indices, in_chans=in_chans,
                pretrained=True, **backbone_kwargs)
            encoder_channels = encoder.feature_info.channels()[::-1]
            self.encoder = encoder
            self.hrtype = 0
        elif "mit" in encoder_name:
            self.encoder = SegformerModel.from_pretrained(
                encoder_name,config=SegformerConfig.from_pretrained(
                    encoder_name, output_hidden_states=True
                ))
            encoder_channels = self.encoder.config.hidden_sizes[::-1]
            self.hrtype = 1

        if "res" in lrbackbone:
            lrencoder = create_model(
                lrbackbone, features_only=True, out_indices=backbone_indices, in_chans=in_chans,
                pretrained=True, **backbone_kwargs)
            lr_chan = lrencoder.feature_info.channels()[-1]
            self.lrencoder = lrencoder
            self.lrtype = 0
        elif "mit" in lrbackbone:
            self.lrencoder = SegformerModel.from_pretrained(
                lrbackbone,config=SegformerConfig.from_pretrained(
                    lrbackbone, output_hidden_states=True
                ))
            lr_chan = self.lrencoder.config.hidden_sizes[-1]
            self.lrtype = 1

        if encoder_channels[0] > 1024:
            self.fusionblock = Conv2dBnAct(
                lr_chan + encoder_channels[0], encoder_channels[0] // 2, kernel_size=(1, 1))
            encoder_channels[0] //= 2
        else:
            self.fusionblock = Conv2dBnAct(
                lr_chan + encoder_channels[0], encoder_channels[0], kernel_size=(1, 1))

        initialize_decoder(self.fusionblock)

        self.lrscale = lrscale

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels[:len(encoder_channels)],
            final_channels=classes,
            norm_layer=norm_layer,
        )

    def forward(self, x, lr):
        if self.hrtype == 0:
            x = self.encoder(x)
        elif self.hrtype == 1:
            x = list(self.encoder(x).hidden_states)
        x.reverse()

        if self.lrtype == 0:
            lr = self.lrencoder(lr)[-1]
        elif self.lrtype == 1:
            lr = self.lrencoder(lr).last_hidden_state        

        ylen, xlen = lr.shape[-2:]
        centorlr = lr[..., ylen//2-ylen//2//self.lrscale-1:ylen//2+ylen//2//self.lrscale+1, \
                      xlen//2-xlen//2//self.lrscale-1:xlen//2+xlen//2//self.lrscale+1]
        centorlr = F.interpolate(centorlr, scale_factor=self.lrscale, mode="bilinear", align_corners=False)
        centorlr = centorlr[..., self.lrscale:centorlr.shape[-2]-self.lrscale,\
                            self.lrscale:centorlr.shape[-1]-self.lrscale]

        x[0] = self.fusionblock(torch.concat((x[0], centorlr), dim=1))

        predmask = self.decoder(x)
        if self.hrtype == 1:
            predmask = F.interpolate(predmask, scale_factor=2, mode="bilinear", align_corners=False)

        return predmask
