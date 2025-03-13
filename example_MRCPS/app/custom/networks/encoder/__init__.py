from .mit import mit
from .convnext import ConvNext
from .timm import timm

modeldict = {
    # hugging face
    "nvidia/mit-b1" : mit,
    "microsoft/swinv2-tiny-patch4-window8-256" : ConvNext,
    # timm
    "resnest26d" : timm,
    "resnest50d" : timm,
    "resnet50d" : timm,
    "efficientnet_b3" : timm,
    # torchvision
}

import sys
if "torchvision.models.swin_t" in sys.modules:
    from networks.encoder.swint import swint
    swin_t = {
        "swinv2t" : swint,
        "swint" : swint
    }
    modeldict = modeldict.update(swin_t)

def encoderfactory(name):
    return modeldict[name](name)