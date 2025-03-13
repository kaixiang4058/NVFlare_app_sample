from torch import nn
from transformers import SegformerModel, SegformerConfig

from .base import base

class mit(base):
    def __init__(self, name):
        super().__init__()
        self.encoder = SegformerModel.from_pretrained(
                name,config=SegformerConfig.from_pretrained(
                    name, output_hidden_states=True
                ))

    def forward(self, inputs):
        return list(self.encoder(inputs).hidden_states[-4:])

    def hidden_size(self):
        return self.encoder.config.hidden_sizes
