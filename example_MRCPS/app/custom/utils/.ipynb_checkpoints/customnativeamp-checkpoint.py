# Copyright The PyTorch Lightning team.
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
from typing import Any, Callable, Optional, Union

import torch
from torch.nn import Module
from torch.optim import LBFGS, Optimizer

import pytorch_lightning as pl
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities import GradClipAlgorithmType

from pytorch_lightning.plugins import MixedPrecisionPlugin
# from pytorch_lightning.plugins import NativeMixedPrecisionPlugin
class CustomNativeMixedPrecisionPlugin(MixedPrecisionPlugin):
# class CustomNativeMixedPrecisionPlugin(NativeMixedPrecisionPlugin):
    """Plugin for Native Mixed Precision (AMP) training with ``torch.autocast``.

    Args:
        precision: Whether to use ``torch.float16`` (``16``) or ``torch.bfloat16`` (``'bf16'``).
        device: The device for ``torch.autocast``.
        scaler: An optional :class:`torch.cuda.amp.GradScaler` to use.
    """

    backend = "native"

    def __init__(
        self, precision: Union[str, int], device: str, scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> None:
        super().__init__(precision, device, scaler)

    def optimizer_step(
        self,
        model: Union["pl.LightningModule", Module],
        optimizer: Optimizer,
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        if self.scaler is None:
            # skip scaler logic, as bfloat16 does not require scaler
            return super().optimizer_step(model, optimizer, optimizer_idx, closure, **kwargs)
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException(
                f"Native AMP and the LBFGS optimizer are not compatible (optimizer {optimizer_idx})."
            )
        closure_result = closure()
        # `unscale` after the closure is executed but before the `on_before_optimizer_step` hook.
        self.scaler.unscale_(optimizer)
        self._after_closure(model, optimizer, optimizer_idx)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if not isinstance(model, pl.LightningModule) or not model.automatic_optimization or not skipped_backward:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            step_output = self.scaler.step(optimizer, **kwargs)
            self.scaler.update()
            return step_output
        return closure_result

    def _after_closure(
        self, model: Union["pl.LightningModule", Module], optimizer: Optimizer, optimizer_idx: int
    ) -> None:
        """Utility to share some code after the closure has been run."""
        if not isinstance(model, pl.LightningModule):
            # none of this applies to Lite
            return
        trainer = model.trainer
        assert trainer is not None
        trainer._call_callback_hooks("on_before_optimizer_step", optimizer, optimizer_idx)
        trainer._call_lightning_module_hook("on_before_optimizer_step", optimizer, optimizer_idx)
        # TODO: this is done for the entire model but should be changed to per-optimizer
        if optimizer_idx == 0:
            self._track_grad_norm(trainer)
        self._clip_gradients(
            model,
            optimizer,
            optimizer_idx,
            clip_val=0.5,
            gradient_clip_algorithm="norm",
        )
    def _clip_gradients(
        self,
        model: Union["pl.LightningModule", Module],
        optimizer: Optimizer,
        optimizer_idx: int,
        clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[GradClipAlgorithmType] = None,
    ) -> None:
        if not isinstance(model, pl.LightningModule):
            return
        model.configure_gradient_clipping(
            optimizer,
            optimizer_idx,
            gradient_clip_val=clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )