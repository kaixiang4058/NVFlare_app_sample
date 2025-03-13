
import numpy as np
import torch 
import random

from dataset.datamodule import DataModule
from model_ch import model_load


from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from utils.customnativeamp import CustomNativeMixedPrecisionPlugin

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


if __name__=='__main__':
    #--parameter
    local_epochs = 10

    #--datamodule
    data_argpath = './cfgs/data_config.yaml'
    data_module = DataModule(data_argpath)

    # train_loader = data_module.train_dataloader()
    # val_loader = data_module.val_dataloader()
    # test_loader = data_module.test_dataloader()

    #--test datamodule
    # for d_i, train_data in enumerate(train_loader['unlabel']):
    #     image, lrimage = train_data
    #     print(image.shape)
    #     if d_i>10:
    #         break



    #--model build
    MODEL_CONF_PATH = './cfgs/traincfg_mrcpsmix_mrmix.yaml'      # model config path
    model, model_config = model_load("MRCPSMixModel",MODEL_CONF_PATH)

    trainer = Trainer(
        max_epochs=local_epochs,
        # callbacks=callbacks,
        log_every_n_steps=50,
        accelerator='gpu',
        devices=-1,
        strategy=DDPStrategy(find_unused_parameters=True),
        plugins=CustomNativeMixedPrecisionPlugin(           \
            model_config['expset']['precision'], "cuda") if     \
            model_config['expset']['precision'] == 16 else None,
        enable_checkpointing=False,
        )


    #validate before training
    model.evaRecords_init()
    model.evaRecord={}
    # accuracy = self.trainer.validate(model=self.model, datamodule=self.valid_loader)
    accuracy = trainer.validate(model=model, datamodule=data_module)

    print('*****valid result*****')
    print(accuracy)



    #--lighting training 
    model.evaRecord={}
    trainer.fit(model=model, datamodule=data_module)



    model.evaRecords_init()
    model.evaRecord={}
    # accuracy = self.trainer.validate(model=self.model, datamodule=self.valid_loader)
    accuracy = trainer.validate(model=model, datamodule=data_module)

    print('*****valid result*****')
    print(accuracy)

        