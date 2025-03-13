# Copyright (c) 2021-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Source of SimpleCNN and moderateCNN: https://github.com/IBM/FedMA/blob/master/model.py,
# SimpleCNN is also from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# MIT License

# Copyright (c) 2020 International Business Machines

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import random
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
import torch

def seed_everything(seed: int):
    import random, os
    import numpy as np    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class MyUnet(nn.Module):
    def __init__(self, num_class=2, lr=0.001, checkpoint_path = None):
        super().__init__()
        seed_everything(12)
        self.model = smp.Unet(
            encoder_name="efficientnet-b3",
            in_channels=3,
            classes=num_class,
        )
        if checkpoint_path != None:
            # checkpoint = "/home/u3346003/work/monai/provision/workspace/secure_project/prod_00/admin@nvidia.com/transfer/Tumor_FedOPT/custom/weights/NCKH12last_epoch=99-global_step=0.ckpt"
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.apply(self.initialize)
            # self.model.load()
        
        # self.model.apply(self.initialize)
        # self.model.load()
    def forward(self, x):
        return self.model(x)

    def initialize(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    