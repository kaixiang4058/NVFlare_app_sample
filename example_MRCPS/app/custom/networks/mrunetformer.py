import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model
from .module import Conv2dBnAct, MScenterMLP
from .initialize import initialize_decoder
from .decoders import UnetDecoder

from transformers import SegformerModel, SegformerConfig

class MRUnetFormer(nn.Module):
    def __init__(
            self,
            encoder_name='resnest26d',
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
        encoder = create_model(
            encoder_name, features_only=True, out_indices=backbone_indices, in_chans=in_chans,
            pretrained=True, **backbone_kwargs)
        # reverse channels [2048, 1024, 512, 256, 64]
        encoder_channels = encoder.feature_info.channels()[::-1]
        self.encoder = encoder

        self.lrencoder = SegformerModel.from_pretrained(
            lrbackbone,config=SegformerConfig.from_pretrained(
                lrbackbone, output_hidden_states=True
            ))
        # reverse channels [512, 320, 128, 64]

        config = self.lrencoder.config
        self.mscenter_mlp = MScenterMLP(config.hidden_sizes, config.decoder_hidden_size, lrscale)
        initialize_decoder(self.mscenter_mlp)

        self.fusionblock = Conv2dBnAct(
            config.decoder_hidden_size * 4 + encoder_channels[0], encoder_channels[0] // 2, kernel_size=(1, 1))
        encoder_channels[0] //= 2
        initialize_decoder(self.fusionblock)

        self.lrscale = lrscale

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=classes,
            norm_layer=norm_layer,
        )

    def forward(self, x, lr, need_fp=False):
        _, _, h, w = x.shape
        x = self.encoder(x)
        x.reverse()
        lr = list(self.lrencoder(lr).hidden_states)
        centerlr = self.mscenter_mlp(lr)
        
        x[0] = self.fusionblock(torch.cat((x[0], centerlr), dim=1))

        if need_fp:
            with torch.no_grad():
                out = self.decoder(x)

            for i in range(len(x)):
                x[i] = nn.Dropout2d(0.5)(x[i])
            #     x[i] = torch.cat((x[i], nn.Dropout2d(0.5)(x[i])))
            out_fp = self.decoder(x)
            # outs = self.decoder(x)
            # out, out_fp = outs.chunk(2)

            return out, out_fp

        predmask = self.decoder(x)

        return predmask


