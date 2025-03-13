from torch import nn
from timm import create_model

from .base import base

class timm(base):
    def __init__(self, name):
        super().__init__()
        self.encoder = create_model(name, features_only=True, pretrained=True)

    def forward(self, inputs):
        return self.encoder(inputs)

    def hidden_size(self):
        return self.encoder.feature_info.channels()
