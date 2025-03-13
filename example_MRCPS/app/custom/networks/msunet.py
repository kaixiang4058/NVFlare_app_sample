import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model
from .module import Conv2dBnAct, MScenterMLP
from .initialize import initialize_decoder, initialize_head
from .decoders import UnetDecoder

from transformers import SegformerModel, SegformerConfig

class MSUnet(nn.Module):
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

        if "swin" in encoder_name:
            from networks.swint import SwinTFeatureExtractor
            encoder = SwinTFeatureExtractor(encoder_name)
            encoder_channels = encoder.in_channels_size()
        else:
            encoder = create_model(
                encoder_name, features_only=True, out_indices=backbone_indices, in_chans=in_chans,
                pretrained=True, **backbone_kwargs)
            encoder_channels = encoder.feature_info.channels()
        self.encoder = encoder

        if "swin" in lrbackbone:
            from networks.swint import SwinTFeatureExtractor
            lrencoder = SwinTFeatureExtractor(lrbackbone)
            lrencoder_channels = lrencoder.in_channels_size()
        else:
            lrencoder = create_model(
                lrbackbone, features_only=True, out_indices=backbone_indices, in_chans=in_chans,
                pretrained=True, **backbone_kwargs)
            lrencoder_channels = lrencoder.feature_info.channels()
        self.lrencoder = lrencoder

        self.mscenter_mlp = MScenterMLP(lrencoder_channels[-4:], decoder_hidden_size=256, lrscale=lrscale)
        initialize_decoder(self.mscenter_mlp)

        if encoder_channels[-1] > 1024:
            self.fusionblock = Conv2dBnAct(
                1024 + encoder_channels[-1], encoder_channels[-1] // 2, kernel_size=(1, 1))
            encoder_channels[-1] //= 2
        else:
            self.fusionblock = Conv2dBnAct(
                1024 + encoder_channels[-1], encoder_channels[-1], kernel_size=(1, 1))

        initialize_decoder(self.fusionblock)

        self.lrscale = lrscale

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels[::-1],
            decoder_channels=decoder_channels[:len(encoder_channels)],
            final_channels=classes,
            norm_layer=norm_layer,
        )

    def forward(self, x, lr):
        _, _, h, w = x.shape

        x = self.encoder(x)
        x.reverse()

        lr = self.lrencoder(lr)[-4:]
        centerlr = self.mscenter_mlp(lr)

        x[0] = self.fusionblock(torch.concat((x[0], centerlr), dim=1))

        predmask = self.decoder(x)
        predmask = F.interpolate(predmask, size=(h,w), mode="bilinear", align_corners=False)

        return predmask
