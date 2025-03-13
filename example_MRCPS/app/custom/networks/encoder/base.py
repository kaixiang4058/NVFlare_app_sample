from torch import nn

class base(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        raise NotImplementedError("Need overwrited.")
    def hidden_size(self):
        raise NotImplementedError("Need overwrited.")
