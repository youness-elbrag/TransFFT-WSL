# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from model import model_interface , TransFPEG
from Dataset import data_interface , dataloader
import argparse
from config import Config

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--gpus', default = [0])
    parser.add_argument('--Epochs', default =50)
    args = parser.parse_args()
    return args

#setup the outline module loading 
model = TransFFTMIL(n_classes=2,Positional_encoding=FFTPEG)
data_module = Interface_DataLoader(random_label_df,path)
path ="./rsna-breast-cancer-detection/train_images/"

def main(Config):

    Model = BrastCancer_Pipline(model)
    data_module = Interface_DataLoader(random_label_df,path)

    tb_logger =TensorBoardLogger(save_dir=Config.Save_Checkpoint,
                                                name =Config.name_version, version =Config.number_Version,
                                                log_graph = True, default_hp_metric = False)
    early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=Config.patience,
            verbose=True,
            mode='min'
        )
    Callbacks =ModelCheckpoint(monitor = 'val_loss',
                                            dirpath = str(Config.log_path_val),
                                            filename = '{epoch:02d}-{val_loss:.4f}',
                                            save_last = True,
                                            save_top_k = 1,
                                            mode = 'min',
                                            save_weights_only = True)
    #---->Instantiate Trainer
    trainer = pl.Trainer(
            plugins=DDPPlugin(find_unused_parameters=False),
            num_sanity_val_steps=0, 
            logger=tb_logger,
            callbacks=[Callbacks,early_stop_callback],
            max_epochs= Config.Epochs,
            gpus=[0],
            amp_level=Config.amp_level,  
            precision=Config.precision,  
            accumulate_grad_batches=Config.grad_acc,
            deterministic=True,
            check_val_every_n_epoch=1,
        )
    #---->train or test
    if args.stage == 'train':
        trainer.fit(model = model, datamodule =data_module)  
    else:
        print("missing arguments")      

if __name__ == '__main__':

    #update config 
    Config.Epochs = args.Epochs
    Config.device_gpu = args.gpu

    main(Config)
