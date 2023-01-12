from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

def split_data_Train_Val(label_csv, val_split=0.30):
    train_set , val_set = train_test_split(label_csv,test_size=val_split)
    return train_set , val_set

class Interface_DataLoader(pl.LightningDataModule):
    def __init__(self, label_csv_rand ,path: str):
        super().__init__()
        self.path = path 
        self.batch_size=Config.batch_size
        self.num_workers = Config.num_workers
        self.train_set , self.val_set= split_data_Train_Val(label_csv_rand,val_split=0.30)
        self.transform_train= transforms.Compose([
             transforms.ToPILImage(),
             transforms.Resize([2559, 1024]),
             transforms.RandomAffine(degrees=(-5, 5), 
                                     translate=(0, 0.05), 
                                     scale=(0.9, 1.1)),   
             transforms.ToTensor(),
             #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

    ])

    def setup(self, stage: str):
    # transforms for images
        if stage == "fit":
            self.training_set = BreastCancer_(self.path,self.train_set,transform=self.transform_train)
            self.validation_set = BreastCancer_(self.path,self.val_set,transform=None)

       

    def train_dataloader(self):
        return  DataLoader(self.training_set,batch_size=self.batch_size,num_workers=self.num_workers ,pin_memory=True,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_set,batch_size=self.batch_size ,num_workers=self.num_workers,pin_memory=True,shuffle=False)    

