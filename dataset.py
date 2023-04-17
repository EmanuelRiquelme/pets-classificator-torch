import torch
from torch.utils.data import Dataset
import os
from  PIL import Image
from torchvision import transforms

class pets(Dataset):
    def __init__(self, root_dir = 'images/',transform = False):
        self.root_dir = root_dir
        self.labels = self.__getlabels__()
        self.transform = transform if transform else transforms.Compose([
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Resize((192,192)),
                transforms.RandomHorizontalFlip(p=.3),
                transforms.Normalize(mean = (.5,.5,.5),std = (.5,.5,.5)),
            ])


    def __getlabels__(self):
        files = [file for file in os.listdir(self.root_dir)]
        files = [file.split('.')[0] for file in files]
        files = [file.split('_')[:-1] for file in files]
        files = ['_'.join(file) for file in files]
        labels = list(set(files))
        return dict(zip(labels,torch.arange(len(labels))))

    def __getfiles__(self):
        return [f'{self.root_dir}{file}' for file in os.listdir(self.root_dir)]

    def __len__(self):
        return len(self.__getfiles__())

    def __getitem__(self,idx):
        file_name = self.__getfiles__()[idx]
        img = self.transform(Image.open(file_name))
        label =  ('_').join(file_name.split('.')[0].split('/')[-1].split('_')[:-1])
        return img,self.labels[label]
