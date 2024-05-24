#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from torch.utils.data.sampler import Sumax_split_size_mb
from convmixer import AdapConvMixer # import from curr dir
from torchvision.transforms import ToTensor, ToPILImage
import torch.optim as optim
from train import training, validate # import from curr dir
from result2mat import Result2Mat # import from curr dir
import h5py # import from curr dir
import matplotlib.pyplot as plt
from train import batch_PSNR # import from curr dir
from dl import PreparePatches, PreparevalPatches # import from curr dir


def main():
   
    dim = 64
    depth = 1 # done 1 so it run fast
    kernel_size = 5
    patch_size = 1 # doubt # changed for testing
    
    train_patches = PreparePatches()
    print(len(train_patches))
    tr_dataloader = DataLoader(train_patches, batch_size=10, shuffle=True, num_workers=0) #self.batch size nhi use ho rha,num workers kya hota hai
    
    val_patches = PreparevalPatches()
    print(len(val_patches))
    val_dataloader = DataLoader(val_patches)

    #100 epochs-BGU , 300-Harvard, 500-Cave
    # epochs = 100
    epochs = 1 # changed for testing
    model_name = 'august'
    checkpoint_dir = './data/BGU/checkpoints/'
    model_path = os.path.join(checkpoint_dir, ''.join([model_name, '-{}-{}.pth']))
    #model_path = os.path.join(checkpoint_dir, ''.join([model_name, '.pth']))
    best_model_path, best_psnr = '', -np.inf
    # best_model_path = model_path.format(dim, depth, kernel_size, patch_size)
###################################   MODEL    ##################################################
    
    model = AdapConvMixer(dim,depth,kernel_size,patch_size)
    #model = AdapConvMixer(dim,depth,kernel_size,patch_size)
    #print(model)
    # model.cuda()
#####################################  LOSS-FUNCTION #############################################
    
    loss_fn = nn.L1Loss()
    loss_fn_mse = nn.MSELoss()
    # loss_fn.cuda()
    
##################################### OPTIMIZER #################################################
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6, betas=(0.9, 0.999))

#################################### SCHEDULAR #################################################
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(20, 100, 20)), gamma=0.5)

#################################### Train ####################################################
    
    train_psnr = []
    val_psnr = []
    test_psnr = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")

        train_epoch_psnr = training(model,loss_fn,  optimizer, tr_dataloader, epoch)
        train_psnr.append(train_epoch_psnr)
        print('train psnr',train_epoch_psnr)
        
        val_epoch_psnr = validate(model,loss_fn_mse, val_dataloader)
        val_psnr.append(val_epoch_psnr)
        print('val_epoch_psnr',val_epoch_psnr)
        
        # plt.plot(val_psnr,label='val')
        # plt.plot(train_psnr,label='train')
        
    #saving best psnr 
        if val_epoch_psnr > best_psnr:
                best_psnr = val_epoch_psnr
                best_model_path = model_path.format(epoch, val_epoch_psnr)
                torch.save(model.state_dict(), best_model_path)
                print('Best PSNR: {:4f}'.format(best_psnr))
    
    
        
####################################### Result 2 mat ############################################################
    
    test_epoch_psnr = Result2Mat(model, val_dataloader, val_patches, best_model_path)
    print('test psnr',test_epoch_psnr)
    test_psnr.append(test_epoch_psnr)
    
                
if __name__ == '__main__':
    main()