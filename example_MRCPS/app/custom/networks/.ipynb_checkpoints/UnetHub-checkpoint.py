import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import encoderfactory
from .module import Conv2dBnAct, MScenterMLP
from .initialize import initialize_decoder
from .decoders import UnetDecoder

class UnetHub(nn.Module):
    def __init__(
            self,
            encoder_name='resnest26d',
            lrbackbone='resnest26d',
            lrscale=8,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=2,
            norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        self.encoder = encoderfactory(encoder_name)
        encoder_channels = self.encoder.hidden_size()

        initialize_decoder(self.encoder)

        self.lrscale = lrscale
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels[::-1],
            decoder_channels=decoder_channels[:len(encoder_channels)],
            final_channels=classes,
            norm_layer=norm_layer,
        )

    def forward(self, x, need_fp=False):
        _, _, h, w = x.shape

        x = self.encoder(x)
        x.reverse()

        if need_fp:
            for i in range(len(x)):
                x[i] = nn.Dropout2d(0.5)(x[i])

            out_fp = self.decoder(x)
            out_fp = F.interpolate(out_fp, size=(h, w), mode="bilinear", align_corners=False)

            return out_fp

        predmask = self.decoder(x)
        predmask = F.interpolate(predmask, size=(h,w), mode="bilinear", align_corners=False)

        return predmask
    


class CentroidUnetHub(nn.Module):
    def __init__(
            self,
            encoder_name='resnest26d',
            lrbackbone='resnest26d',
            lrscale=8,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=2,
            norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        self.encoder = encoderfactory(encoder_name)
        encoder_channels = self.encoder.hidden_size()
        initialize_decoder(self.encoder)

        # self.lrencoder = encoderfactory(lrbackbone)
        # lrencoder_channels = self.lrencoder.hidden_size()

        # self.center_mlp = MScenterMLP(encoder_channels[-4:], decoder_hidden_size=256, lrscale=lrscale)
        # initialize_decoder(self.center_mlp)

        # if encoder_channels[-1] > 1024:
        #     self.fusionblock = Conv2dBnAct(
        #         1024 + encoder_channels[-1], encoder_channels[-1] // 2, kernel_size=(1, 1))
        #     encoder_channels[-1] //= 2
        # else:
        #     self.fusionblock = Conv2dBnAct(
        #         1024 + encoder_channels[-1], encoder_channels[-1], kernel_size=(1, 1))

        # initialize_decoder(self.fusionblock)


        self.lrscale = lrscale

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels[::-1],
            decoder_channels=decoder_channels[:len(encoder_channels)],
            final_channels=classes,
            norm_layer=norm_layer,
        )

    def forward(self, x, need_fp=False):
        _, _, h, w = x.shape

        x = self.encoder(x)
        x.reverse()

        # lr = self.lrencoder(lr)[-4:]
        # centerlr = self.mscenter_mlp(lr)

        # x[0] = self.fusionblock(torch.concat((x[0], centerlr), dim=1))

        if need_fp:
            for i in range(len(x)):
                x[i] = nn.Dropout2d(0.5)(x[i])

            out_fp = self.decoder(x)
            out_fp = F.interpolate(out_fp, size=(h, w), mode="bilinear", align_corners=False)

            return out_fp

        predmask = self.decoder(x)
        predmask = F.interpolate(predmask, size=(h,w), mode="bilinear", align_corners=False)

        #if is tissue branch output with centroid part

        mid_x = int(w/2)
        mid_y = int(h/2)
        lr_w = int(w/self.lrscale)
        lr_h = int(h/self.lrscale)
        predmask_centroid = predmask[:,:,mid_y-lr_h:mid_y+lr_h, mid_x-lr_w:mid_x+lr_w]

        return predmask, predmask_centroid

    