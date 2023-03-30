# Packages
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary

# Local files
import data
import encoder
import decoder
import autoencoder
import training


import numpy as np
import argparse
import json

# Random Seed
torch.manual_seed(0)

from torchvision.utils import make_grid

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        with torch.no_grad():
          for t, m, s in zip(tensor, self.mean, self.std):
              t.mul_(s).add_(m)
              # The normalize code -> t.sub_(m).div_(s)
          return tensor

def peek_results(dataloader, model, bottle_neck, exp_name):
  unorm = UnNormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
  for img, lb in dataloader:
    fig, ax = plt.subplots(figsize=(16,8))
    fig1, ax1 = plt.subplots(figsize=(16,8))
    ax.set_xticks([]); ax.set_yticks([])
    ax1.set_xticks([]); ax1.set_yticks([])
    img_output = model(img.cuda()).cpu()
    ax.imshow(make_grid(unorm(img_output), nrow=16).permute(1,2,0))
    fig.savefig(f"{exp_name}_{bottle_neck}")
    ax1.imshow(make_grid(img, nrow=16).permute(1,2,0))
    fig1.savefig(f"{exp_name}_{bottle_neck}_OG")
    break

def main(bottle_neck, batch_size = 64, experiment_type = 1, optimizer = "Adam", lr = 0.001):
    #Load data
    train_ds, val_ds, test_ds = data.get_datasets()
    dataloader_train = data.dataset_to_dataloader(train_ds,batch_size=batch_size)
    dataloader_test = data.dataset_to_dataloader(test_ds,batch_size=batch_size, shuffle= False)
    dataloader_val = data.dataset_to_dataloader(val_ds,batch_size=batch_size)
    ## Bottle neck size
    bn = bottle_neck

    if experiment_type == 1:
        ## Trained from Scratch
        encoder_ = encoder.resnet18(pretrained=False,num_classes=bn)
        decoder_ = decoder.ResNet18Dec(z_dim=bn)
        exp_name = f"AE_{bn}"

    elif experiment_type == 2: 
        ## Trained from frozen weights
        encoder_ = encoder.resnet18(pretrained=True,num_classes=bn)
        decoder_ = decoder.ResNet18Dec(z_dim=bn)
        for param in encoder_.parameters():
            param.requires_grad = False

        encoder_.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        encoder_.fc = nn.Linear(in_features=512, out_features=bn, bias=True)
        exp_name = f"FrozenAE_{bn}"
    
    else: 
        ## Trained from pretrained init weights
        encoder_ = encoder.resnet18(pretrained=True,num_classes=bn)
        decoder_ = decoder.ResNet18Dec(z_dim=bn)
        for param in encoder_.parameters():
            param.requires_grad = False

        encoder_.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        encoder_.fc = nn.Linear(in_features=512, out_features=bn, bias=True)
        exp_name = f"Pretrained_Init_AE_{bn}"


    criterion = nn.MSELoss()
    model = autoencoder.AutoEncoder(encoder_, decoder_)
    ## set the optimizer
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    #move to GPU
    if torch.cuda.is_available():
        model.to(torch.device('cuda:0'))

    ## Call the training 
    model, hist = training.train(model,
            criterion,
            optimizer,
            dataloader_train,
            dataloader_val,
            save_file_name=f'History_{exp_name}_{bn}.pt',
            max_epochs_stop = 5,
            n_epochs = 30,
            print_every = 1)
    
    ## Save training time and loss
    hist.to_csv(f'History_{exp_name}_{bn}.csv')
    
    ## Save a image for OG and the output
    peek_results(dataloader_test, model, bn, exp_name)

main(bottle_neck=8)
    



