import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig
from typing import Optional
from .split_attn import SplAtConv2d
from .mix_transformer import SegFormerBlock

class Conv2dBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_channels) if norm_layer is not None else nn.Identity()

        self.act = act_layer() if act_layer == nn.GELU else \
            act_layer(inplace=True) if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2.0, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=1, norm_layer=norm_layer, act_layer=act_layer)
        self.scale_factor = scale_factor
        self.conv1 = Conv2dBnAct(in_channels, out_channels, **conv_args)
        self.conv2 = Conv2dBnAct(out_channels, out_channels, **conv_args)

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        if self.scale_factor != 1.0:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SFDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_act_layer=nn.ReLU, conv_norm_layer=nn.BatchNorm2d,
                num_heads=8, sr_ratio=8, att_norm_layer=nn.LayerNorm, drop_path_rate:list=[0.,0.], att_depth=2):
        super().__init__()
        if len(drop_path_rate) != att_depth:
            raise ValueError(f"Inconsistent between drop_path {len(drop_path_rate)} and att_depth {att_depth}")

        conv_args = dict(kernel_size=3, padding=1, norm_layer=conv_norm_layer, act_layer=conv_act_layer)
        att_args = dict(num_heads=num_heads, sr_ratio=sr_ratio, norm_layer=att_norm_layer)
        
        self.conv1 = Conv2dBnAct(in_channels, out_channels, **conv_args)
        
        self.AttBlocks = nn.ModuleList([SegFormerBlock(dim=out_channels, drop_path=dpr, **att_args)
                                        for dpr in drop_path_rate])
        self.norm = att_norm_layer(out_channels)

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        if skip is not None:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear')
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv1(x)

        B, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        for blk in self.AttBlocks:
            x = blk(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x




class DecodeSplAtBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        conv_args = dict(kernel_size=3, norm_layer=norm_layer, act_layer=act_layer)
        self.conv1 = Conv2dBnAct(
            in_channels=in_channels, out_channels=out_channels, padding=1, **conv_args)
        self.conv2 = SplAtConv2d(
            in_channels=out_channels, out_channels=out_channels, **conv_args)
        self.act = act_layer() if act_layer == nn.GELU else \
            act_layer(inplace=True) if act_layer is not None else nn.Identity()

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        if skip is not None:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear')
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.act(x)

        return x

class DecodeSplAtBlockv2(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        conv_args = dict(norm_layer=norm_layer, act_layer=act_layer)
        self.conv1 = Conv2dBnAct(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, **conv_args)
        self.conv2 = SplAtConv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, **conv_args)
        self.conv3 = SplAtConv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, **conv_args)
        act = act_layer() if act_layer == nn.GELU else \
            act_layer(inplace=True) if act_layer is not None else nn.Identity()
        self.act1 = act
        self.act2 = act

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        if skip is not None:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear')
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')

        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.act1(x)
        x = self.conv3(x) + x
        x = self.act2(x)

        return x

class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, input_dim, decoder_hidden_size=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states

class MScenterMLP(nn.Module):
    def __init__(self, hidden_sizes, decoder_hidden_size, lrscale):
        super().__init__()
        self.linear_c = nn.ModuleList()
        for input_dim in hidden_sizes:
            mlp = SegformerMLP(input_dim=input_dim, decoder_hidden_size=decoder_hidden_size)
            self.linear_c.append(mlp)
        
        self.lrscale = lrscale

    def forward(self, lr: torch.Tensor):
        B, _, H, W = lr[-1].size()

        hidden_states = ()
        for centerlr, mlp in zip(lr, self.linear_c):
            ylen, xlen = centerlr.shape[-2:]
            centerlr = mlp(centerlr)
            centerlr = centerlr.permute(0, 2, 1)
            centerlr = centerlr.reshape(B, -1, ylen, xlen)

            centerlr = centerlr[..., ylen//2-ylen//2//self.lrscale-1:ylen//2+ylen//2//self.lrscale+1, \
                        xlen//2-xlen//2//self.lrscale-1:xlen//2+xlen//2//self.lrscale+1]
            centerlr = F.interpolate(centerlr, scale_factor=self.lrscale, mode="bilinear", align_corners=False)
            centerlr = centerlr[..., self.lrscale:centerlr.shape[-2]-self.lrscale,\
                                self.lrscale:centerlr.shape[-1]-self.lrscale]
            centerlr = F.interpolate(
                    centerlr, size=(H, W), mode='area')
            hidden_states += (centerlr,)

        return torch.cat(hidden_states, dim=1)

# Different for the first low-level feature due to the upsampling interpolate,
# and floating point different on e-8
class MScenterMLPv2(nn.Module):
    def __init__(self, config: SegformerConfig, lrscale):
        super().__init__()
        self.linear_c = nn.ModuleList()
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            self.linear_c.append(mlp)
        
        self.lrscale = lrscale

    def forward(self, lr: torch.Tensor):
        B, _, H, W = lr[-1].size()

        hidden_states = ()
        for centerlr, mlp in zip(lr, self.linear_c):
            ylen, xlen = centerlr.shape[-2:]
            centerlr = mlp(centerlr)
            centerlr = centerlr.permute(0, 2, 1)
            centerlr = centerlr.reshape(B, -1, ylen, xlen)

            scaleratio = H // (ylen // self.lrscale)
            centerlr = centerlr[..., ylen//2-ylen//2//self.lrscale-1:ylen//2+ylen//2//self.lrscale+1, \
                        xlen//2-xlen//2//self.lrscale-1:xlen//2+xlen//2//self.lrscale+1]
            centerlr = F.interpolate(centerlr, scale_factor=scaleratio, mode="bilinear", align_corners=False)
            centerlr = centerlr[..., scaleratio:centerlr.shape[-2]-scaleratio,\
                                scaleratio:centerlr.shape[-1]-scaleratio]

            hidden_states += (centerlr,)

        return torch.cat(hidden_states, dim=1)