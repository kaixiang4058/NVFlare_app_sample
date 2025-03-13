from transformers import ConvNextModel

from .base import base

class ConvNext(base):
    def __init__(self, name):
        super().__init__()
        self.encoder = ConvNextModel.from_pretrained(name, output_hidden_states=True)

    def forward(self, inputs):
        return self.encoder(inputs).hidden_states[1:]

    def hidden_size(self):
        # [768, 384, 192, 96]
        return self.encoder.config.hidden_sizes
