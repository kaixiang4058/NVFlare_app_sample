from typing import List

import torch
import torch.nn as nn
from functools import partial

from .module import DecoderBlock, SFDecoderBlock
from .initialize import initialize_decoder, initialize_head, initialize_SF

class UnetDecoder(nn.Module):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=2,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            block=DecoderBlock,
            center=False,
    ):
        super().__init__()

        conv_args = dict(norm_layer=norm_layer, act_layer=act_layer)
        if center:
            channels = encoder_channels[0]
            self.center = block(channels, channels, scale_factor=1.0, **conv_args)
        else:
            self.center = nn.Identity()

        in_channels = [in_chs + skip_chs for in_chs, skip_chs in zip(
            [encoder_channels[0]] + list(decoder_channels[:-1]),
            list(encoder_channels[1:]) + [0])]
        out_channels = decoder_channels

        self.blocks = nn.ModuleList()
        for in_chs, out_chs in zip(in_channels, out_channels):
            if out_chs == decoder_channels[-1]:
                self.blocks.append(DecoderBlock(in_chs, out_chs, **conv_args))
            else:
                self.blocks.append(block(in_chs, out_chs, **conv_args))

        self.final_conv = nn.Conv2d(out_channels[-1], final_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        initialize_decoder(self.blocks)
        initialize_head(self.final_conv)

    def forward(self, x: List[torch.Tensor]):
        encoder_head = x[0]
        skips = x[1:]
        x = self.center(encoder_head)
        for i, b in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = b(x, skip)
        x = self.final_conv(x)
        return x

class UnetDecoderv2(nn.Module):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64),
            final_channels=2,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            block=DecoderBlock,
    ):
        super().__init__()

        conv_args = dict(norm_layer=norm_layer, act_layer=act_layer)

        in_channels = [in_chs + skip_chs for in_chs, skip_chs in zip(
            [encoder_channels[0]] + list(decoder_channels[:-1]),
            list(encoder_channels[1:]))]
        out_channels = decoder_channels

        self.blocks = nn.ModuleList()
        for in_chs, out_chs in zip(in_channels, out_channels):
            self.blocks.append(block(in_chs, out_chs, **conv_args))

        self.final_conv = nn.Conv2d(out_channels[-1], final_channels, kernel_size=1)

        initialize_decoder(self.blocks)
        initialize_head(self.final_conv)

    def forward(self, x: List[torch.Tensor]):
        skips = x[1:]
        x = x[0]
        for i, b in enumerate(self.blocks):
            x = b(x, skips[i])

        x = self.final_conv(x)
        return x

class UFormerDecoder(nn.Module):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64),
            final_channels=2,
            conv_act_layer=nn.ReLU,
            conv_norm_layer=nn.BatchNorm2d,
            block=SFDecoderBlock,
            num_heads=(4, 2, 1),
            sr_ratios=(2, 4, 8),
            att_depths=(2, 2, 2),
            drop_path_rate=0.08,
            att_norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        conv_args = dict(conv_norm_layer=conv_norm_layer, conv_act_layer=conv_act_layer)

        in_channels = [in_chs + skip_chs for in_chs, skip_chs in zip(
            [encoder_channels[0]] + list(decoder_channels[:-1]),
            list(encoder_channels[1:]))]
        out_channels = decoder_channels

        # assign drop path rate
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, sum(att_depths))]  # stochastic depth decay rule
        cur = 0
        dpr_chunk = []
        for d in att_depths:
            dpr_chunk.append(dpr[cur:cur+d])
            cur += d

        self.blocks = nn.ModuleList()
        for in_chs, out_chs, nheads, sr, depth, dpr in \
            zip(in_channels, out_channels, num_heads, sr_ratios, att_depths, dpr_chunk):
            att_args = dict(
                num_heads=nheads, sr_ratio=sr, att_depth=depth, \
                att_norm_layer=att_norm_layer, drop_path_rate=dpr)
            self.blocks.append(block(in_chs, out_chs, **conv_args, **att_args))

        self.final_conv = nn.Conv2d(out_channels[-1], final_channels, kernel_size=1)

        initialize_decoder(self.blocks)
        initialize_head(self.final_conv)

    def forward(self, x: List[torch.Tensor]):
        skips = x[1:]
        x = x[0]
        for i, b in enumerate(self.blocks):
            x = b(x, skips[i])

        x = self.final_conv(x)
        return x
