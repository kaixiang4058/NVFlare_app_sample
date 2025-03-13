import copy
import os
import pyvips
import numpy as np
import torch
import torch.optim as optim
from networks.unet import MyUnet, seed_everything
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType
from nvflare.app_common.pt.pt_fedproxloss import PTFedProxLoss

from tqdm import tqdm
import yaml
from pathlib import Path
from utils.datamodule import DataModule
from utils.metrics import binary_mean_iou

class LiverTumorLearner(Learner):
    def __init__(
        self,
        data_dir,
        data_seed,
        val_freq: int =1,
        epochs: int = 1,
        lr: float = 1e-2,
        fedproxloss_mu: float = 0.0,
        central: bool = False,
        analytic_sender_id: str = "analytic_sender",
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        """Simple CIFAR-10 Trainer.
        Args:
            train_idx_root: directory with site training indices for CIFAR-10 data.
            aggregation_epochs: the number of training epochs for a round. Defaults to 1.
            lr: local learning rate. Float number. Defaults to 1e-2.
            fedproxloss_mu: weight for FedProx loss. Float number. Defaults to 0.0 (no FedProx).
            central: Bool. Whether to simulate central training. Default False.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component.
                If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
            batch_size: batch size for training and validation.
            num_workers: number of workers for data loaders.
        Returns:
            a Shareable with the updated local model after running `execute()`
            or the best local model depending on the specified task.
        """
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point
        # self.config_file =  config_file
        self.data_dir = data_dir
        self.data_seed = data_seed
        self.validation_freq = val_freq
        self.aggregation_epochs = epochs
        self.lr = lr
        self.fedproxloss_mu = fedproxloss_mu
        self.best_acc = 0.0
        self.central = central
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.writer = None
        self.analytic_sender_id = analytic_sender_id

        # Epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0

        # following will be created in initialize() or later
        self.app_root = None
        self.client_id = None
        self.local_model_file = None
        self.best_local_model_file = None
        self.best_global_model_file = None
        self.writer = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.criterion_prox = None
        self.transform_train = None
        self.transform_valid = None
        self.train_dataset = None
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.data_module = None
        self.test_loader = None
        self.global_best = 0.0

    def initialize(self, parts: dict, fl_ctx: FLContext):
        """
        Note: this code assumes a FL simulation setting
        Datasets will be initialized in train() and validate() when calling self._create_datasets()
        as we need to make sure that the server has already downloaded and split the data.
        """

        # when the run starts, this is where the actual settings get initialized for trainer
        
        
        # Set the paths according to fl_ctx
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        print(f"--------------------- {self.client_id} ---------------------")
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized at \n {self.app_root} \n with args: {fl_args}",
        )

        self.local_model_file = os.path.join(self.app_root, "local_model.pt")
        self.best_local_model_file = os.path.join(self.app_root, "best_local_model.pt")
        self.best_global_model_file = os.path.join(self.app_root, "best_global_model.pt")

        # Select local TensorBoard writer or event-based writer for streaming
        self.writer = parts.get(self.analytic_sender_id)  # user configured config_fed_client.json for streaming
        if not self.writer:  # use local TensorBoard writer only
            self.writer = SummaryWriter(self.app_root)

        # different hospital config yaml files
        # yaml_file_names = [
        #     'config_ncku.yaml',
        #     'config_kvgh.yaml'
        # ]
        
        # # Get Config Dictionary
        # if self.client_id =='site-1':
        #     self.config_file = os.path.join(self.config_file,yaml_file_names[0])
        # if self.client_id =='site-2':
        #     self.config_file = os.path.join(self.config_file,yaml_file_names[1])
        # print("---------------------------------------- config file --------------------------------------\n", self.config_file)
        
        # self.configs = yaml.safe_load(Path(self.config_file).read_text())
        # set the training-related parameters
        # can be replaced by a config-style block
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.model = MyUnet(checkpoint = self.pretrain_weight)
        self.model = MyUnet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay= 5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()

    def _create_datasets(self, fl_ctx: FLContext):
        """To be called only after Cifar10DataSplitter downloaded the data and computed splits"""
        """See DataModule.py and Dataset.py for more details about dataset and dataloaders"""
        if self.data_module is None:
            seed_everything(self.data_seed)
            self.data_module = DataModule(self.data_dir, batch_size = self.batch_size)
            self.data_module.setup()
        
        self.train_loader = self.data_module.train_dataloader()
        self.valid_loader = self.data_module.val_dataloader()
        self.test_loader = self.data_module.test_dataloader()

    def finalize(self, fl_ctx: FLContext):
        # collect threads, close files here
        pass

    def _local_train(self, fl_ctx, train_loader, abort_signal: Signal, val_freq: int = 0):
        for epoch in range(self.aggregation_epochs):
            if abort_signal.triggered:
                return
            self.model.train()
            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch
            self.log_info(fl_ctx, f"Local epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})")
            avg_loss = 0.0
            for i, (inputs, labels) in tqdm(enumerate(train_loader)):
                if abort_signal.triggered:
                    return
                img, label_mask = inputs.to(self.device), labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(img)
                loss = self.criterion(outputs, label_mask)

                loss.backward()
                self.optimizer.step()
                current_step = epoch_len * self.epoch_global + i
                avg_loss += loss.item()
            self.writer.add_scalar("train_loss", avg_loss / len(train_loader), current_step)
            if val_freq > 0 and epoch % val_freq == 0:
                acc = self._local_valid(self.valid_loader, abort_signal, tb_id="val_acc_local_model", fl_ctx=fl_ctx)
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.save_model(is_best=True)
        del loss, outputs

    def save_model(self, is_best=False):
        # save model
        model_weights = self.model.state_dict()
        save_dict = {"model_weights": model_weights, "epoch": self.epoch_global}
        if is_best:
            save_dict.update({"best_acc": self.best_acc})
            torch.save(save_dict, self.best_local_model_file)
        else:
            torch.save(save_dict, self.local_model_file)
    
    def save_global_model(self):
        # save model
        model_weights = self.model.state_dict()
        save_dict = {"model_weights": model_weights, "epoch": self.epoch_global}
        save_dict.update({"best_acc": self.best_acc})
        torch.save(save_dict, self.best_global_model_file)

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) : # -> Shareable
        self._create_datasets(fl_ctx)

        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
        
        print(f'################################ Current at [{current_round +1 } / {total_rounds}] rounds ##############################################')

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        
        # Ensure data kind is weights.
        if not dxo.data_kind == DataKind.WEIGHTS:
            self.log_error(fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # global_weights = dxo.data
        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        # local_var_dict = self.model.state_dict()
        # model_keys = global_weights.keys()
        # for var_name in local_var_dict:
        #     if var_name in model_keys:
        #         # weights = global_weights[var_name]
        #         weights = torch.as_tensor(global_weights[var_name], device=self.device)
        #         try:
        #             # reshape global weights to compute difference later on
        #             # global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
        #             local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
        #             # update the local dict
        #             # local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
        #         except BaseException as e:
        #             raise ValueError(f"Convert weight from {var_name} failed") from e
                
        # local_var_dict = self.model.state_dict()
        # model_keys = global_weights.keys()
        # n_loaded = 0
        # for var_name in local_var_dict:
        #     if var_name in model_keys:
        #         weights = torch.as_tensor(global_weights[var_name], device=self.device)
        #         try:
        #             # update the local dict
        #             local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
        #             n_loaded += 1
        #         except BaseException as e:
        #             raise ValueError(f"Convert weight from {var_name} failed") from e
        # self.model.load_state_dict(local_var_dict)
        # if n_loaded == 0:
        #     raise ValueError(f"No weights loaded for validation! Received weight dict is {global_weights}")
        # self.model.load_state_dict(local_var_dict)
        
        torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
        self.model.load_state_dict(torch_weights)
        
        # ---- test before training ------ check pretrained weights
        if current_round==0:
            acc = self._local_valid(self.valid_loader, abort_signal, tb_id="val_acc_local_model", fl_ctx=fl_ctx)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"---------------------Pretrained val_acc_local_model: {acc:.4f} --------------------")

        # local steps
        epoch_len = len(self.train_loader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")
        print(f"Local steps per epoch: {epoch_len}")

        # local train
        self._local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_loader,
            abort_signal=abort_signal,
            val_freq=1 if self.central else 0,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.aggregation_epochs

        # perform valid after local train
        acc = self._local_valid(self.valid_loader, abort_signal, tb_id="val_acc_local_model", fl_ctx=fl_ctx)
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"--------------------- val_acc_local_model: {acc:.4f} --------------------")

        # save model
        self.save_model(is_best=False)
        if acc > self.best_acc:
            self.best_acc = acc
            self.save_model(is_best=True)

        new_weights = self.model.state_dict()
        new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}
        print("New Weights type!!: ",type(new_weights))
        # print(new_weights.items())
        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=new_weights,
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: epoch_len},
        )
        return outgoing_dxo.to_shareable()
    
    def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) :#-> Shareable
        # Retrieve the best local model saved during training.
        if model_name == ModelName.BEST_MODEL:
            model_data = None
            try:
                # load model to cpu as server might or might not have a GPU
                model_data = torch.load(self.best_local_model_file, map_location="cpu")
            except BaseException as e:
                raise ValueError("Unable to load best model") from e

            # Create DXO and shareable from model data.
            if model_data:
                # convert weights to numpy to support FOBS
                model_weights = model_data["model_weights"]
                for k, v in model_weights.items():
                    model_weights[k] = v.numpy()
                dxo = DXO(data_kind=DataKind.WEIGHTS, data=model_weights)
                return dxo.to_shareable()
            else:
                # Set return code.
                self.log_error(fl_ctx, f"best local model not found at {self.best_local_model_file}.")
                return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
        else:
            raise ValueError(f"Unknown model_type: {model_name}")  # Raised errors are caught in LearnerExecutor class.

    def _local_valid(self, valid_loader, abort_signal: Signal, tb_id=None, fl_ctx=None):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            iou_metric, total = 0, 0
            for _i, (inputs, labels) in enumerate(valid_loader):
                if abort_signal.triggered:
                    return None
                img, label_mask = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(img)
                label_mask = label_mask.type(torch.cuda.IntTensor)
                iou = binary_mean_iou(torch.softmax(outputs.detach(), dim=1), label_mask)

                total += img.data.size()[0]
                iou_metric += iou
            metric = iou_metric / float(len(valid_loader))
            metric = round(metric.cpu().numpy()*100,2)
            if tb_id:
                self.writer.add_scalar(tb_id, metric, self.epoch_global)
        return metric

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) :
        self._create_datasets(fl_ctx)

        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get validation information
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
        model_owner = shareable.get(ReservedHeaderKey.HEADERS).get(AppConstants.MODEL_OWNER)
        if model_owner:
            self.log_info(fl_ctx, f"Evaluating model from {model_owner} on {fl_ctx.get_identity_name()} data")
        else:
            model_owner = "global_model"  # evaluating global model during training

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(global_weights[var_name], device=self.device)
                try:
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
                    n_loaded += 1
                except BaseException as e:
                    raise ValueError(f"Convert weight from {var_name} failed") from e
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(f"No weights loaded for validation! Received weight dict is {global_weights}")

        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            # perform valid before local train
            global_acc = self._local_valid(self.valid_loader, abort_signal, tb_id="val_acc_global_model", fl_ctx=fl_ctx)
            if global_acc>self.global_best:
                self.global_best = global_acc
                self.save_global_model()
                
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f" ---------------- val_acc_global_model ({model_owner}): {global_acc} ----------------")

            return DXO(data_kind=DataKind.METRICS, data={MetaKey.INITIAL_METRICS: global_acc}, meta={}).to_shareable()

        elif validate_type == ValidateType.MODEL_VALIDATE:
            # perform valid
            train_acc = self._local_valid(self.train_loader, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f" ---------------- training acc ({model_owner}): {train_acc} ----------------")

            val_acc = self._local_valid(self.valid_loader, abort_signal)
            
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f" ---------------- validation acc ({model_owner}): {val_acc} ----------------")


            test_acc = self._local_valid(self.test_loader, abort_signal)

            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"---------------- test acc ({model_owner}): {test_acc} ----------------")
            
            self.log_info(fl_ctx, "Evaluation finished. Returning shareable")

            val_results = {"train_accuracy": train_acc, "val_accuracy": val_acc, "test_accuracy": test_acc}

            metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
            return metric_dxo.to_shareable()

        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)
