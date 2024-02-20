import torch.nn.functional as F
import torch
import random
import torch.nn as nn
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import random_split
from torch.utils.data import DataLoader,Dataset
from glob import glob
import albumentations
from albumentations.pytorch import ToTensorV2



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(323)

class CancerDataset(Dataset):

    def __init__(self,base_dir,label_dir,transform=None):
        self.base_dir =base_dir
        self.list_files =glob(os.path.join(self.base_dir,"*.tif"))
        self.transform =transform

        df = pd.read_csv(label_dir)
        key = []
        value = []
        for i in df["id"]:
            key.append(i)
        for j in df["label"]:
            value.append(j)
        self.dic = dict(zip(key, value))

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        image_path =self.list_files[index]
        img =cv2.imread(image_path)
        color_image =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            transform =self.transform(image=color_image)
            color_image = transform['image']
        temp = []
        temp.append(image_path.split("/")[-1])
        imgname = temp[0].split(".")[0]
        self.imgnage =imgname
        return color_image,self.dic[imgname]

data_transforms = albumentations.Compose([
    albumentations.Resize(cfg['image_size'], cfg['image_size']),
    albumentations.RandomRotate90(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.Flip(p=0.5),
    albumentations.OneOf([
        albumentations.CLAHE(clip_limit=2), albumentations.IAASharpen(), albumentations.IAAEmboss(),
        albumentations.RandomBrightness(), albumentations.RandomContrast(),
        albumentations.JpegCompression(), albumentations.Blur(), albumentations.GaussNoise()], p=0.5),
    albumentations.HueSaturationValue(p=0.5),
    albumentations.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ToTensorV2()
    ])
