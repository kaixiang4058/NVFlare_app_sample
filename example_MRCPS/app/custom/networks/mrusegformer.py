import torch
from torch import nn
import torch.nn.functional as F
from transformers import ConvNextModel
from transformers import SegformerModel, SegformerConfig

# from transformers import Swinv2ForImageClassification

from .module import Conv2dBnAct, DecoderBlock, MScenterMLP
from .initialize import initialize_decoder
from .decoders import UnetDecoder

class MRUSegFormer(nn.Module):
    def __init__(
            self,
            encoder_name='nvidia/mit-b1',
            lrbackbone='nvidia/mit-b1',
            classes=2,
            lrscale=8
    ):
        super().__init__()

        if 'mit' in encoder_name:
            self.encoder = SegformerModel.from_pretrained(
                encoder_name,config=SegformerConfig.from_pretrained(
                    encoder_name, output_hidden_states=True
                ))
        elif 'convnext' in encoder_name:
            self.encoder = ConvNextModel.from_pretrained(encoder_name, output_hidden_states=True)
        self.lrscale = lrscale

        if 'mit' in lrbackbone:
            self.lrencoder = SegformerModel.from_pretrained(
                lrbackbone,config=SegformerConfig.from_pretrained(
                    lrbackbone, output_hidden_states=True
                ))
        elif 'convnext' in lrbackbone:
            self.lrencoder = ConvNextModel.from_pretrained(lrbackbone, output_hidden_states=True)
        
        # reverse channels [512, 320, 128, 64]
        hidden_sizes = self.encoder.config.hidden_sizes[::-1]

        config = self.lrencoder.config
        self.mscenter_mlp = MScenterMLP(config.hidden_sizes, config.decoder_hidden_size, lrscale)
        initialize_decoder(self.mscenter_mlp)

        self.fusionblock = Conv2dBnAct(
            config.decoder_hidden_size * 4 + hidden_sizes[0], hidden_sizes[0], kernel_size=(1, 1))

        initialize_decoder(self.fusionblock)

        self.decoder = UnetDecoder(
            encoder_channels=hidden_sizes,
            decoder_channels=(256, 128, 64, 32),
            final_channels=classes,
            norm_layer=nn.BatchNorm2d,
            block=DecoderBlock
        )

    def forward(self, inputs, lrinputs, need_fp=False):
        _, _, h, w = inputs.shape

        x = list(self.encoder(inputs).hidden_states[-4:])
        x.reverse()
        lr = list(self.lrencoder(lrinputs).hidden_states[-4:])
        centerlr = self.mscenter_mlp(lr)

        x[0] = self.fusionblock(torch.cat((x[0], centerlr), dim=1))

        if need_fp:
            with torch.no_grad():
                out = self.decoder(x)
                out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
            for i in range(len(x)):
                x[i] = nn.Dropout2d(0.5)(x[i])
            #     x[i] = torch.cat((x[i], nn.Dropout2d(0.5)(x[i])))

            out_fp = self.decoder(x)
            out_fp = F.interpolate(out_fp, size=(h, w), mode="bilinear", align_corners=False)
            # out, out_fp = outs.chunk(2)

            return out, out_fp

        predmask = self.decoder(x)
        predmask = F.interpolate(predmask, size=(h, w), mode="bilinear", align_corners=False)

        return predmask

