import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model
from .module import Conv2dBnAct, MScenterMLP
from .initialize import initialize_decoder
from .decoders import UnetDecoder

from transformers import SegformerModel, SegformerConfig

class MSUTransCNN(nn.Module):
    def __init__(
            self,
            encoder_name="nvidia/mit-b1",
            lrbackbone="resnest26d",
            lrscale=8,
            backbone_kwargs=None,
            backbone_indices=None,
            decoder_channels=(256, 128, 64, 32),
            in_chans=3,
            classes=2,
            norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        # NOTE some models need different backbone indices specified based on the alignment of features
        # and some models won't have a full enough range of feature strides to work properly.
        self.encoder = SegformerModel.from_pretrained(
            encoder_name,config=SegformerConfig.from_pretrained(
                encoder_name, output_hidden_states=True
            ))
        hidden_sizes = self.encoder.config.hidden_sizes[::-1]
        # reverse channels [512, 320, 128, 64]
        
        lrencoder = create_model(
            lrbackbone, features_only=True, out_indices=backbone_indices, in_chans=in_chans,
            pretrained=True, **backbone_kwargs)
        # reverse channels [2048, 1024, 512, 256, 64]
        lrencoder_channels = lrencoder.feature_info.channels()
        self.lrencoder = lrencoder

        self.mscenter_mlp = MScenterMLP(lrencoder_channels[-4:], decoder_hidden_size=256, lrscale=lrscale)
        initialize_decoder(self.mscenter_mlp)

        self.fusionblock = Conv2dBnAct(
            256 * 4 + hidden_sizes[0], hidden_sizes[0], kernel_size=(1, 1))
        
        initialize_decoder(self.fusionblock)

        self.lrscale = lrscale

        self.decoder = UnetDecoder(
            encoder_channels=hidden_sizes,
            decoder_channels=decoder_channels,
            final_channels=classes,
            norm_layer=norm_layer,
        )

    def forward(self, x, lr):
        _, _, h, w = x.shape

        x = list(self.encoder(x).hidden_states[-4:])
        x.reverse()

        lr = self.lrencoder(lr)[-4:]
        centerlr = self.mscenter_mlp(lr)

        x[0] = self.fusionblock(torch.concat((x[0], centerlr), dim=1))

        predmask = self.decoder(x)
        predmask = F.interpolate(predmask, size=(h,w), mode="bilinear", align_corners=False)

        return predmask

