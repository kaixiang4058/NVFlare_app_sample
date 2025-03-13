import torch
import os
import numpy as np
import copy
import time
import random  

from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable

from torch.utils.tensorboard import SummaryWriter


    
class PersonalLearner(Learner):
    def __init__(
        self,
        val_freq: int =1,
        epochs: int = 1,
        data_seed: int = 42,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
    ):
        #self, data_argPath:str, model_confPath:str):
        super().__init__()

        # ---FL used---
        # FL work task name (preserve)
        self.train_task_name = train_task_name
        self.submit_model_task_name = submit_model_task_name

        # accuracy
        self.best_acc=0
        self.global_best=0

        #model output record
        self.eva_records = []   
        self.loss_records = []

        #epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0


        # ---Personal parameter setting (personal code)---
        '''
        #local epoch and valid freq (personal)
        self.local_epochs = epochs
        self.val_freq = val_freq

        #seed paramter (personal)
        self.data_seed = data_seed
        '''

    def initialize(self, parts: dict, fl_ctx: FLContext):
        # ---FL used---
        #FL: get client root info withfl_ctx
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.cfg_root = os.path.join(self.app_root, 'custom/')
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized at \n {self.app_root} \n with args: {fl_args}",
        )
        
        #模型權重存取路徑
        self.local_model_file = os.path.join(self.app_root, "local_model.pt")
        self.best_local_model_file = os.path.join(self.app_root, "best_local_model.pt")
        self.best_global_model_file = os.path.join(self.app_root, "best_global_model.pt")

        # ---Personal parameter setting (personal code)---
        '''
        #Set seed
        random.seed(self.data_seed)
        torch.manual_seed(self.data_seed)
        torch.cuda.manual_seed(self.data_seed)

        #data prepare
        data_argpath = os.path.join(self.cfg_root, data_config_path)
        self.dataloader = DataloaderPrepare(data_argpath)

        #model prepare
        model_argpath = os.path.join(self.cfg_root, self.model_config_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.model_config = model_load("MRCPSMixModel", model_argpath, self.cfg_root)
        self.model = self.model.to(self.device)
        '''

    def finalize(self, fl_ctx: FLContext):
        #loss_types = ["loss1", "loss2", ..., "total"]
        loss_records_arr = np.array(self.loss_records)    #if error watch out data number (because insufficeint epoch number will influence result)
        np.save(os.path.join(self.app_root,f'loss_record.npy'),loss_records_arr)
    
        #eva_types = ["iou", "f1", ...]
        eva_records_arr = np.array(self.eva_records)
        np.save(os.path.join(self.app_root,f'eva_records.npy'),eva_records_arr)
        

    #==== Personal function ====
    def local_train(self, fl_ctx, abort_signal: Signal):
        if abort_signal.triggered:
            return
        print('========start local training========')
        self.log_info(fl_ctx, f"Local epoch {self.client_id}: 0 (lr={self.lr})")

        epoch = 0
        self.epoch_global = self.epoch_of_start_time + epoch

        # ---personal model training step (personal code)---
        '''
        model.train()
        for epo in self.epoch:
            for step, (data, label) in enumerate(dataloader*):
                pre = model(data)
                loss = criterion(pre, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        '''

        # ---record train loss each epoch---
        self.loss_records.append('new epoch loss record')
        
    def local_valid(self, abort_signal: Signal, tb_id=None, fl_ctx=None, stage=None):
        if abort_signal.triggered:
            return
        print('========start validing========')
        
         # ---personal model validation step (personal code)---
        '''
        model.eval()
        for step, (data, label) in enumerate(dataloader*):
            pre = model(data)
            accuracy = criterion(pre, label)
        '''
        # ---return valid result (not only train step used)---
        if stage=='train':
            self.eva_records.append(copy.deepcopy(self.model.evaRecords_load()))    #存取所有驗證準確
        
        return 'new epoch evaluation record'    #回傳過程僅須提供單一 accuracy數值 (用以取得最佳權重使用)
    
    def save_model(self, is_best=False):
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


    #---main workflow---
    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) :
        # ---FL used---
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        print("====== load local_variable_dict =======")
        time_rc = time.time()
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                except Exception as e:
                    raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
        self.model.load_state_dict(local_var_dict)

        # # make a copy of model_global as reference for potential FedProx loss or SCAFFOLD
        # model_global = copy.deepcopy(self.model)
        # for param in model_global.parameters():
        #     param.requires_grad = False
        # print('====== load local_variable_dict cost time:', time.time()-time_rc)


        # ---local train---
        print(f"===== local train  {self.epoch_of_start_time} =====")
        time_rc = time.time()
        self.local_train(
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.local_epochs

        # local steps record
        epoch_len = len(self.data_module.train_dataloader()['unlabel'])
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")
        print(f'===== local train {self.epoch_of_start_time} cost time:', time.time()-time_rc)


        # ---valid after local train---
        print(f"===== local val {self.epoch_of_start_time} =====")
        time_rc = time.time()
        acc = self.local_valid(abort_signal, tb_id="val_acc_local_model", fl_ctx=fl_ctx, stage='train')
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"val_acc_local_model: {acc:.4f}")
        print(f'=== local val {self.epoch_of_start_time} cost time:', time.time()-time_rc)

        # ---save model---
        print(f"===== save model {self.epoch_of_start_time}  =====")
        self.save_model(is_best=False)
        if acc > self.best_acc:
            self.save_model(is_best=True)


        # ---FL transfer model---
        # compute delta model, global model has the primary key set
        print(f"===== delta model {self.epoch_of_start_time} =====")
        local_weights = self.model.state_dict()
        model_diff = {}
        for name in global_weights:
            if name not in local_weights:
                continue
            model_diff[name] = local_weights[name].cpu().numpy() - global_weights[name]
            if np.any(np.isnan(model_diff[name])):
                self.system_panic(f"{name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # build the shareable
        print(f"===== share model {self.epoch_of_start_time}  =====")
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo.to_shareable()
    

    def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) -> Shareable:
        # Retrieve the best local model saved during training.
        if model_name == ModelName.BEST_MODEL:
            model_data = None
            try:
                # load model to cpu as server might or might not have a GPU
                model_data = torch.load(self.best_local_model_file, map_location="cpu")
            except Exception as e:
                self.log_error(fl_ctx, f"Unable to load best model: {e}")

            #to numpy
            model_weights = {k: v.numpy() for k, v in model_data["model_weights"].items()}
            
            # Create DXO and shareable from model data.
            if model_data:
                dxo = DXO(data_kind=DataKind.WEIGHTS, data=model_weights)
                return dxo.to_shareable()
            else:
                # Set return code.
                self.log_error(fl_ctx, f"best local model not found at {self.best_local_model_file}.")
                return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
        else:
            raise ValueError(f"Unknown model_type: {model_name}")  # Raised errors are caught in LearnerExecutor class.

    
    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) :
        '''
        self:log_info, model, local_valid(), valid_loader
        '''
        # ---FL used---
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get validation information
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
        model_owner = shareable.get(ReservedHeaderKey.HEADERS).get(AppConstants.MODEL_OWNER)
        if model_owner:
            self.log_info(fl_ctx, f"Evaluating model from {model_owner} on {fl_ctx.get_identity_name()}")
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
                except Exception as e:
                    raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(f"No weights loaded for validation! Received weight dict is {global_weights}")

        # ---valid work (before train, ---
        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            # perform valid before local train
            global_acc = self.local_valid(abort_signal, tb_id="val_acc_global_model", fl_ctx=fl_ctx)
            if global_acc>self.global_best:
                self.global_best = global_acc
                self.save_global_model()
            
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_acc_global_model ({model_owner}): {global_acc}")

            return DXO(data_kind=DataKind.METRICS, data={MetaKey.INITIAL_METRICS: global_acc}, meta={}).to_shareable()

        elif validate_type == ValidateType.MODEL_VALIDATE:
            val_acc = self.local_valid(abort_signal)
        
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"validation acc ({model_owner}): {val_acc}")
            self.log_info(fl_ctx, "Evaluation finished. Returning shareable")

            # val_results = {"train_accuracy": train_acc, "val_accuracy": val_acc}
            val_results = { "val_accuracy": val_acc}

            metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
            return metric_dxo.to_shareable()

        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)

    
    
        