from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
import pydicom as dc
import pandas as pd 
from utilits import *

class BreastCancer_(Dataset):
    def __init__(self,root_dir,label_csv , transform=None):
        self.img_dir = root_dir
        self.label = label_csv
        self.transform = transform
        
    def __len__(self):
        return len(self.label["image_id"])
    
    def __getitem__(self,idx):
        id_ = self.label.iloc[idx]
        image_id = str(id_.patient_id) +"/"+ str(id_.image_id)
        img_path = os.path.join(self.img_dir,image_id) + ".dcm"
        img_array = dc.dcmread(img_path).pixel_array
        label = id_.cancer
        
        if self.transform:
            ## we implamet to method crop and Equalization 
            (x,y,w,h) = crop_coords(img_array)
            crop_img = img_array[y:y+h,x:x+w]
            equalizer = histogram_equalization(crop_img)
            image = normalization_(equalizer)
            #image_= self.transform(image)
            label = torch.tensor(label)
            return  self.transform(image), label
        else:     
            img_array = normalization_(img_array) 
            image_ = cv2.resize(img_array, (1024, 2559)).astype(np.float32)
            image_ = np.expand_dims(image_,axis=0)
            image_ = torch.tensor(image_)
            label = torch.tensor(label)
            return image_ , label     