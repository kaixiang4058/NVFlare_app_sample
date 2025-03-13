import torch
import os
import numpy as np
import copy
import time

from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable

from torch.utils.tensorboard import SummaryWriter

from dataset.datamodule import DataModule
from model_ch import model_load
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from utils.customnativeamp import CustomNativeMixedPrecisionPlugin

# import torch.distributed as dist

# # if dist.is_initialized():
# #     dist.destroy_process_group()
# # Set environment variables explicitly

# # if "RANK" not in os.environ:
# os.environ["RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
# os.environ["MASTER_ADDR"] = "127.0.0.1"
# # os.environ["MASTER_PORT"] = "35659"
    
# import socket
# def find_free_port():
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind(("", 0))
#         return s.getsockname()[1]
# os.environ["MASTER_PORT"] = str(find_free_port())
# # print('-----------------------', str(find_free_port()))

# # Initialize distributed process group
# if not dist.is_initialized():
#     # dist.init_process_group(backend="gloo", init_method="env://")
#     dist.init_process_group(backend="nccl", init_method="env://")

# if dist.is_initialized():
#     print(f"Backend: {dist.get_backend()}")
#     print(f"Rank: {dist.get_rank()}")
#     print(f"World Size: {dist.get_world_size()}")
    
class MRCPSLearner(Learner):
    def __init__(
        self,
        data_config= './cfgs/data_config.yaml',
        data_seed=42,
        val_freq: int =1,
        epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
        lr: float = 1e-2,
        fedproxloss_mu: float = 0.0,
        central: bool = False,
        model_config_file = './cfgs/traincfg_mrcpsmix_mrmix.yaml',
        analytic_sender_id: str = "analytic_sender",
    ):
        #self, data_argPath:str, model_confPath:str):
        super().__init__()
        #data parameter
        self.data_config_file = data_config
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        #seed paramter
        self.data_seed = data_seed

        #epoch and valid freq
        self.local_epochs = epochs
        self.val_freq = val_freq

        #train task name
        self.train_task_name = train_task_name
        self.submit_model_task_name = submit_model_task_name

        #loss info (not used in lighting train)
        self.lr = lr
        self.fedproxloss_mu = fedproxloss_mu
        self.central = central

        # accuracy
        self.best_acc=0
        self.global_best=0

        #model setting parameter
        self.model_config_file = model_config_file # model config etc
        self.eva_records = []   #model output record
        self.loss_records = []

        #tensorboard writer
        self.writer = None
        self.analytic_sender_id = analytic_sender_id

        #epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0


    def seed_everything(self, seed: int):
        import random, os
        import numpy as np    
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def initialize(self, parts: dict, fl_ctx: FLContext):
        """
        Note: this code assumes a FL simulation setting
        Datasets will be initialized in train() and validate() when calling self._create_datasets()
        as we need to make sure that the server has already downloaded and split the data.
        """
        # when the run starts, this is where the actual settings get initialized for trainer
        
        #Set seed
        self.seed_everything(self.data_seed)

        #Set the paths according to fl_ctx
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.cfg_root = os.path.join(self.app_root, 'custom/')
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        print(f"--------------------- {self.client_id} ---------------------")
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized at \n {self.app_root} \n with args: {fl_args}",
        )
        
        temp_dict = {'app_root'}
        

        #self.save_root = ''
        self.local_model_file = os.path.join(self.app_root, "local_model.pt")
        self.best_local_model_file = os.path.join(self.app_root, "best_local_model.pt")
        self.best_global_model_file = os.path.join(self.app_root, "best_global_model.pt")

        #Select local TensorBoard writer or event-based writer for streaming
        self.writer = parts.get(self.analytic_sender_id)  # user configured config_fed_client.json for streaming
        if not self.writer:  # use local TensorBoard writer only
            self.writer = SummaryWriter(self.app_root)

        #data prepare
        data_argpath = os.path.join(self.cfg_root, self.data_config_file)
        print(">>>>>>>>>>>>>",data_argpath)
        self.data_module = DataModule(data_argpath)
        # self.train_loader = self.data_module.train_dataloader()
        # self.valid_loader = self.data_module.val_dataloader()
        # self.test_loader = self.data_module.test_dataloader()

        #model prepare
        model_argpath = os.path.join(self.cfg_root, self.model_config_file)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.model_config = model_load("MRCPSMixModel", model_argpath, self.cfg_root)
        self.model = self.model.to(self.device)

        #trainner prepare
        self.trainer = Trainer(
            max_epochs=self.local_epochs,
            # callbacks=callbacks,
            log_every_n_steps=50,
            accelerator='gpu',
            devices=-1,
            strategy="ddp",
            # strategy=DDPStrategy(find_unused_parameters=True, timeout=600),
            plugins=CustomNativeMixedPrecisionPlugin(           \
                "16-mixed", "cuda") if     \
                self.model_config['expset']['precision'] == 16 else None,
            # precision="bf16-mixed",
            enable_checkpointing=False,
            # num_sanity_val_steps=-1,
            )
        
    def finalize(self, fl_ctx: FLContext):
        print(self.loss_records)
        #loss_types = ["b1_sup", "b2_sup", "b1_cps", "b2_cps", "total"]
        loss_records_arr = np.array(self.loss_records)    #if error watch out data number (because insufficeint epoch number will influence result)
        np.save(os.path.join(self.app_root,f'loss_record.npy'),loss_records_arr)
    
        eva_records_arr = np.array(self.eva_records)
        np.save(os.path.join(self.app_root,f'eva_records.npy'),eva_records_arr)
        # collect threads, close files here
        pass

    #self function
    def reset_trainer(self):
        # --need set with customer method--
        del self.trainer
        # optimizer and initial setup of the training is handled by the trainer init
        self.trainer = Trainer(
                max_epochs=self.local_epochs,
                # callbacks=callbacks,
                log_every_n_steps=50,
                accelerator='gpu',
                devices=-1,
                strategy="ddp",
                # strategy=DDPStrategy(find_unused_parameters=True, timeout=600),
                plugins=CustomNativeMixedPrecisionPlugin(           \
                    "16-mixed", "cuda") if     \
                    self.model_config['expset']['precision'] == 16 else None,
                # precision="bf16-mixed",
                enable_checkpointing=False,
                )
        

    def local_train(self, fl_ctx, abort_signal: Signal):
        if abort_signal.triggered:
            return
        print('========start local training========')
        self.log_info(fl_ctx, f"Local epoch {self.client_id}: 0 (lr={self.lr})")

        epoch = 0
        self.epoch_global = self.epoch_of_start_time + epoch

        self.reset_trainer()

        self.model.evaRecords_init()
        self.model.evaRecord={}
        print(self.trainer)
        self.trainer.fit(model=self.model, datamodule=self.data_module)
        print(self.model.loss_record_epoch)
        # self.eva_records.append(copy.deepcopy(self.model.evaRecords_load()))
        
        epoch_len = len(self.data_module.train_dataloader()['unlabel'])
        print('---------------------')
        # print(self.data_module.train_dataloader()['unlabel'])
        # print(epoch_len)

        current_step = epoch_len * self.epoch_global
        
        loss_types = ["b1_sup", "b2_sup", "b1_cps", "b2_cps", "total"]
        for rec_loss_list in self.model.loss_record_epoch:
            for i in range(len(rec_loss_list)):
                self.writer.add_scalar(loss_types[i], rec_loss_list[i], self.epoch_of_start_time+i) #model no loss
        # print(self.model.evaRecord)
        # self.writer.add_scalar("eva_records", self.eva_records[-1], current_step) #model no loss

        self.eva_records.append(copy.deepcopy(self.model.evaRecords_load()))
        self.loss_records = copy.deepcopy(self.model.loss_record_epoch)
        print(self.loss_records)
        # FedProx loss term
        # if self.fedproxloss_mu > 0:
        #     fed_prox_loss = self.criterion_prox(self.model, model_global)
        #     loss += fed_prox_loss
        
    def local_valid(self, abort_signal: Signal, tb_id=None, fl_ctx=None):
        if abort_signal.triggered:
            return
        
        self.reset_trainer()
        # self.model.evaRecords_init()
        # self.model.evaRecord={}
        # accuracy = self.trainer.validate(model=self.model, datamodule=self.valid_loader)
        accuracy = self.trainer.validate(model=self.model, datamodule=self.data_module)

        print('*****valid result*****')
        print(accuracy)

        # self.eva_records.append(copy.deepcopy(self.model.evaRecords_load()))
        # if tb_id:
        #     self.writer.add_scalar(tb_id, accuracy[-1]['valid ens iou_score'], self.epoch_global)
        print(self.model.evaRecords_load())
        return self.model.evaRecords_load()[0,-1,0]
    

    def save_model(self, is_best=False):
        '''save model'''
        # save model
        model_weights = self.model.state_dict()
        #model_weights = self.fastai_nnet.state_dict()
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

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) :
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
        # print(';;;;;;;;model;;;;;;;;;;')
        # print(self.model.state_dict()['branch1.encoder.encoder.conv1.0.weight'][0])
        # print(self.model.state_dict()['branch2.decoder.final_conv.weight'][0])

        # make a copy of model_global as reference for potential FedProx loss or SCAFFOLD
        model_global = copy.deepcopy(self.model)
        for param in model_global.parameters():
            param.requires_grad = False
        print('====== load local_variable_dict cost time:', time.time()-time_rc)

        # local train
        print(f"===== local train  {self.epoch_of_start_time} =====")
        time_rc = time.time()
        self.local_train(
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.local_epochs

        # local steps
        epoch_len = len(self.data_module.train_dataloader()['unlabel'])
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")
        print(f'===== local train {self.epoch_of_start_time} cost time:', time.time()-time_rc)

        # perform valid after local train
        print(f"===== local val {self.epoch_of_start_time} =====")
        time_rc = time.time()
        acc = self.local_valid(abort_signal, tb_id="val_acc_local_model", fl_ctx=fl_ctx)
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"val_acc_local_model: {acc:.4f}")
        print(f'=== local val {self.epoch_of_start_time} cost time:', time.time()-time_rc)

        # save model
        print(f"===== save model {self.epoch_of_start_time}  =====")
        self.save_model(is_best=False)
        if acc > self.best_acc:
            self.save_model(is_best=True)

        # print(';;;;;;;;model after local train;;;;;;;;;;')
        # print(acc, self.best_acc)
        # print(self.model.state_dict()['branch1.encoder.encoder.conv1.0.weight'][0])
        # print(self.model.state_dict()['branch2.decoder.final_conv.weight'][0])

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
        '''Retrieve the best local model saved during training.'''
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
            # perform valid
            # train_acc = self.local_valid(self.train_dataloader, abort_signal)
            # if abort_signal.triggered:
            #     return make_reply(ReturnCode.TASK_ABORTED)
            # self.log_info(fl_ctx, f"training acc ({model_owner}): {train_acc}")

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

    
    
        