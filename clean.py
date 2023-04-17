import torch
import os
from  PIL import Image
from torchvision import transforms

transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize((192,192)),
        transforms.RandomHorizontalFlip(p=.3),
        transforms.Normalize(mean = (.5,.5,.5),std = (.5,.5,.5)),
                ])

def clean(root_dir = 'images/',transform = transform):
    files =  [f'{root_dir}{file}' for file in os.listdir(root_dir)]
    for file in files:
        try:
            transform(Image.open(file))
        except:
            os.remove(file)

if __name__ == '__main__':
    clean()
