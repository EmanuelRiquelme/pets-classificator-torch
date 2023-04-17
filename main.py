import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from model import Model
from torch.utils.data import DataLoader
import os
from utils import validation,save_model,load_model
from tqdm import trange
from dataset import pets

dataset = pets() 
train_size = round(int(len(dataset)*.8))
test_size = len(dataset)-train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size,test_size])

batch_size= 32
train_set= DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers = 4)
val_set= DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers = 4)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = Model().to(device)
opt = torch.optim.Adam(model.parameters(),lr=0.0001)
loss_fn = nn.CrossEntropyLoss()
epochs = 20

def train_pipeline(train_data = train_set,model = model,
                    loss_fn = loss_fn,opt = opt,epochs = epochs,device = device,val_data = val_set,threshold = .9):
    for epoch in  (t := trange(epochs)):
        it = iter(train_data)
        temp_loss = []
        for _ in range(len(train_data)):
            input,target = next(it)
            input,target = input.to(device),target.to(device)
            opt.zero_grad()
            output = model(input)
            loss = loss_fn(output,target)
            temp_loss.append(loss.item())
            loss.backward()
            opt.step()
        val =validation(val_data,model)
        temp_loss = sum(temp_loss)/len(temp_loss)
        t.set_description(f'validation: {val:.2f},loss : {temp_loss:.2f}')
        if val >= threshold:
            break

if __name__ == '__main__':
    print(f'initial val: {validation(val_set,model):.2f}')
    train_pipeline()
    save_model(model,opt)
