import torch
import torch.nn as nn
from typing import List

from timm import create_model
from networks.decoders import UnetDecoder

class Unet(nn.Module):
    def __init__(self, encoder_name='resnest26d', classes=2, decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__()

        self.encoder = create_model(
            encoder_name, features_only=True, in_chans=3, pretrained=True)

        encoder_channels = self.encoder.feature_info.channels()[::-1]

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=classes,
        )


    def forward(self, inputs, need_fp=False):
        x = self.encoder(inputs)
        x.reverse()

        if need_fp:
            # for i in range(len(x)):
            #     x[i] = torch.cat((x[i], nn.Dropout2d(0.5)(x[i])))

            for i in range(len(x)):
                x[i] = nn.Dropout2d(0.5)(x[i])
            outs = self.decoder(x)

            return outs

            # out, out_fp = outs.chunk(2)

            # return out, out_fp

        x = self.decoder(x)

        return x
