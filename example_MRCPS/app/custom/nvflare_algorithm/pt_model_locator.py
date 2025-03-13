# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import List, Union

import torch.cuda
from nvflare_algorithm.pt_constants import PTConstants

from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import model_learnable_to_dxo
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager
from networks.unet import MyUnet
from model_ch import model_load

class PTModelLocator(ModelLocator):
    def __init__(self):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.checkpoint_path = checkpoint_path
        # self.model = MyUnet()
        # self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.model_config_file = './cfgs/traincfg_mrcpsmix_mrmix.yaml',
        self.cfg_root = os.path.join(self.app_root, 'custom/')
        model_argpath = os.path.join(self.cfg_root, self.model_config_file)
        self.model, self.model_config = model_load("MRCPSMixModel", model_argpath, self.cfg_root)
        # self.checkpoint = torch.load(checkpoint_path, map_location = device)
        # self.model.load_state_dict(self.checkpoint['state_dict'])
        
    def get_model_names(self, fl_ctx: FLContext) -> List[str]:
        return [PTConstants.PTServerName]

    def locate_model(self, model_name, fl_ctx: FLContext) -> Union[DXO, None]:
        self.log_info(
            fl_ctx,
            f'---------------{model_name}----------------------',
        )
        if model_name == PTConstants.PTServerName:
            try:
                server_run_dir = fl_ctx.get_engine().get_workspace().get_app_dir(fl_ctx.get_job_id())
                model_path = os.path.join(server_run_dir, PTConstants.PTFileModelName)
                ## edit
                # model_path = self.checkpoint_path
                if not os.path.exists(model_path):
                    self.log_info(
                        fl_ctx,
                        f"##############NONE############",
                    )
                    print("##############NONE############")
                    return None

                # Load the torch model
                device = "cuda" if torch.cuda.is_available() else "cpu"
                data = torch.load(model_path, map_location=device)
                
                ##edit
                # data = data['state_dict']
                # Set up the persistence manager.
                if self.model:
                    default_train_conf = {"train": {"model": type(self.model).__name__}}
                else:
                    default_train_conf = None

                # Use persistence manager to get learnable
                persistence_manager = PTModelPersistenceFormatManager(data, default_train_conf=default_train_conf)
                ml = persistence_manager.to_model_learnable(exclude_vars=None)

                # Create dxo and return
                return model_learnable_to_dxo(ml)
            except Exception as e:
                self.log_error(fl_ctx, f"Error in retrieving {model_name}: {e}.", fire_event=False)
                return None
        else:
            self.log_error(fl_ctx, f"PTModelLocator doesn't recognize name: {model_name}", fire_event=False)
            return None
