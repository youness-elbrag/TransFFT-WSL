import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl

def SeedEveryThing(seed:int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
def cross_entropy_torch(x, y):
    loss = F.cross_entropy(x,y)
    return loss        
        
Config = Config_Tuning()
SeedEveryThing(Config.seed)

class BrastCancer_Pipline(pl.LightningModule):
    def __init__(self,model,**kargs):
        super(BrastCancer_Pipline,self).__init__()
        ### the PL of model
        self.save_hyperparameters()
        self.load_model()
        ### setup the Optimizer and loss functions
        self.Optimizer = torch.optim.Adam(self.model.parameters(),lr = Config.lr)
        self.loss = nn.CrossEntropyLoss()
        self.n_classes = Config.n_classes
         # simple accuracy computation
        
        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]    
        self.AUROC = torchmetrics.AUROC(task="binary",num_classes=2, average = 'macro')
        metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task="binary",num_classes = 2,
                                                                           average = 'micro'),
                                                     torchmetrics.CohenKappa(task="binary",num_classes = 2),
                                                     torchmetrics.F1Score(task="binary",num_classes = 2,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(task="binary",average = 'macro',
                                                                         num_classes = 2),
                                                     torchmetrics.Precision(task="binary",average = 'macro',
                                                                            num_classes = 2)])
        self.valid_metrics = metrics.clone(prefix = 'val_')
         #--->random
        self.shuffle = True
        self.count = 0
        
    def training_step(self, batch, batch_idx):
        #---->inference
        data, label = batch
        logits, Y_prob, Y_hat = self.model(data=data, label=label)
        #---->loss
        loss = self.loss(logits, label)
        #---->acc log
        Y_hat = int(Y_hat)
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

        return {'loss': loss} 
    
    def training_epoch_end(self, training_step_outputs):
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)] 
        
    #######
    ### here we did the same as Traiing PL we changed only the input_data disttro
    #######
    def validation_step(self, batch, batch_idx):
        data, label = batch
        logits, Y_prob, Y_hat = self.model(data=data, label=label)
        #---->acc log
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    def validation_epoch_end(self, val_step_outputs):
        logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim = 0)
        max_probs = torch.stack([x['Y_hat'] for x in val_step_outputs])
        target = torch.stack([x['label'] for x in val_step_outputs], dim = 0)
        
        #---->
        self.log('val_loss', cross_entropy_torch(logits, target), prog_bar=True, on_epoch=True, logger=True)
        self.log('auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)

        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        #---->random, if shuffle data, change seed
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)  
        
    def configure_optimizers(self):
        #Caution! You always need to return a list here (just pack your optimizer into one :))
        return [self.Optimizer]  
    
    def load_model(self):
        model = self.hparams.model
        self.model = model    