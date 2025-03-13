import torch
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from .base import base

class swint(base):
    def __init__(self, name):
        super().__init__()
        self.encoder = SwinTFeatureExtractor(name)

    def forward(self, inputs):
        return self.encoder(inputs)

    def hidden_size(self):
        return self.encoder.in_channels_size()

class SwinTFeatureExtractor(torch.nn.Module):
    def __init__(self, name="swinv2t"):
        super().__init__()

        if name == "swinv2t":
            m = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
        elif name == "swint":
            m = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)

        self.body = create_feature_extractor(
            m, return_nodes={f'features.{k}': str(v)
                             for v, k in enumerate([1, 3, 5, 7])})

    def in_channels_size(self):
        return [96, 192, 384, 768]

    def forward(self, x):
        outs = self.body(x)
        logits = []

        for key in outs:
            logits.append(torch.permute(outs[key], (0, 3, 1, 2))) 

        return logits


