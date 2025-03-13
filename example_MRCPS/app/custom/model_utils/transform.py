import torch
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

def get_strong_aug():
    return albu.Compose(
        [
        albu.Transpose(),
        albu.RandomRotate90(p=1),
        albu.ColorJitter(hue=0.04, saturation=0.08),
        ], additional_targets={'lrimage': 'image', 'lrmask': 'mask'}
    )

# Dihedral group
def get_weak_aug():
    return albu.Compose(
        [
        albu.Transpose(),
        albu.RandomRotate90(p=1),
        ], additional_targets={'lrimage': 'image', 'lrmask': 'mask'}
    )

def get_preprocess():
    _transform = [
        ToTensorV2(transpose_mask=True),
        albu.Lambda(image=norm_scale),
    ]
    return albu.Compose(
        _transform,
        additional_targets={'lrimage': 'image', 'lrmask': 'mask'}
        )

def get_taki_preprocess():
    return albu.Compose([
                albu.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5), p=1),
                ToTensorV2(),
            ])

def norm_scale(x, **kwargs):
    return x.to(torch.float32) / 255.0

