import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from tqdm import tqdm
from dl import PreparePatches
import matplotlib.pyplot as plt
import scipy.io as scio
import scipy
from scipy.io import savemat
import numpy as np
            
def training(model, loss_fn, optimizer, tr_dataloader, epoch):
    print('Training')
    model.train()
    running_loss = 0.0
    running_loss_2 = 0.0
    psnr_sum=0.0
    counter = 0
    
    for data in tqdm(tr_dataloader):
        counter += 1
        inputs, labels = data
        #interpolate to decide about the size of labels acc. to the patch size - 15/7/'22
        #labels = F.interpolate(labels,size=[64,64])
        # inputs1,labels1 = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)  #changes
        
        # print('outputs in tr',outputs.shape)
        
        
        loss = loss_fn(outputs, labels) #changes
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        outputs2 = outputs.cpu().detach()
        labels2 = labels.cpu().detach()
        # neeche wali line bhi thodi change kari hai
        psnr = batch_PSNR(outputs2,labels2)
        psnr_sum += psnr.item()
    
    psnr_sum = psnr_sum/counter
    print('av psnr',psnr_sum)
    
    epoch_loss = running_loss / counter
    print(f"Train Loss : {epoch_loss:.8f}")

    return psnr_sum 

#use this to calculate psnr(PSNR with data-range) -15/7/'22
def batch_PSNR(im_true, im_fake, data_range=255):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    Ifake = im_fake.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C*H*W)
    psnr = 10. * torch.log((data_range**2)/err) / np.log(10.)
    return torch.mean(psnr)

#no data-range being used here.  - 26/7/'22
# def batch_PSNR(im_true, im_fake, data_range=255):
#     N = im_true.size()[0]
#     C = im_true.size()[1]
#     H = im_true.size()[2]
#     W = im_true.size()[3]
#     Itrue = im_true.clamp(0.,1.).resize_(N, C*H*W)
#     Ifake = im_fake.clamp(0.,1.).resize_(N, C*H*W)
#     mse = nn.MSELoss(reduction='mean')
#     err = mse(Itrue, Ifake)
#     psnr = 10. * torch.log(1/err) / np.log(10.)
#     return psnr


def validate(model, loss_fn_mse, val_dataloader):
    print('Validating')
    model.eval()
    psnr_sum = 0.0
    running_loss_psnr = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_dataloader), total=int(len(val_dataloader)/val_dataloader.batch_size)):
            counter += 1
            inputs,labels = data
            # inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
#change loss function in validate, originally loss = loss_fn(outputs, labels) - 11/7/'22
            # loss = loss_fn(outputs, labels)
            # print('shape of output',outputs.shape)
            # print('shape of labels', labels.shape)
            loss_forpsnr = loss_fn_mse(outputs, labels)
            # print(loss_forpsnr)
            psnr = batch_PSNR(outputs,labels)
    #             if counter % 336 == 0:
    #                 print(psnr)
            psnr_sum += psnr.item()
# validate doesnt require loss,we need psnr here.(use below lines of code for loss cal in validate) - 15/7/'22    
            #running_loss += loss.item()     #use this running loss when your loss fnt is L1 - 18/7/'22
            running_loss_psnr += loss_forpsnr.item()

        #print(counter)
        #epoch_loss = running_loss / counter
    epoch_loss_psnr = running_loss_psnr / counter
    print(f"Val for psnr Loss:{epoch_loss_psnr:.8f}")
    psnr_sum = psnr_sum/counter
    print('av psnr',psnr_sum)
    #return epoch_loss_psnr, psnr_sum
    return psnr_sum
