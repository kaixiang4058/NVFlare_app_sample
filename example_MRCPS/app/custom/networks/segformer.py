import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation

class SegFormer(nn.Module):
    def __init__(self, encoder_name='nvidia/mit-b1', classes=2):
        super().__init__()

        self.segmentor = SegformerForSemanticSegmentation.from_pretrained(
            encoder_name, num_labels=classes)

    def forward(self, inputs):
        y_pred = self.segmentor(inputs)
        y_pred = nn.functional.interpolate(
                y_pred.logits, size=inputs.shape[-2:], mode="bilinear", align_corners=False
            )

        return y_pred

@torch.no_grad()
def test(encoder_name:str, num_labels:int, inputshape:int):
    model = SegFormer(encoder_name, num_labels=num_labels)
    print(f'params: {sum(p.numel() for p in model.parameters())}')
    print(model)
    out = model(torch.rand(1,3,inputshape,inputshape))
    print(f"shape: {out.shape}")

if __name__ == '__main__':
    # van-small, van-tiny
    test("nvidia/mit-b1", 2, 512)
