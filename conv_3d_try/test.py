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
from torch.utils.data import Dataset, DataLoader
from convmixer import Method
import sys
import h5py
import random
class PreparePatches(Dataset):     #load list of path to __init__
    def __init__(self, fil):
        self.h5f = h5py.File(fil,'r')
        self.keys = list(self.h5f.keys())
        #self.keys=list(self.keys[0])
        random.shuffle(self.keys)
    
    def __len__(self):
        return len(self.keys) 
        
    def key(self, index):
        return str(self.keys[index])
    
    def __getitem__(self,index):    #load each image/file in __getitem__
        key = str(self.keys[index])
        data = np.array(self.h5f[key])
        data = torch.Tensor(data)
        # print('printing shape of data', data.shape)
        inputs = torch.cat([data[0:31, :, :], torch.unsqueeze(data[30, :, :], 0)], dim = 0)
        labels = data[31:62, :, :]
        # print('inputs in PreparePateches', inputs.shape)
        # print('labels in PreparePateches', labels.shape)
        # return data[0:31,:,:], data[31:62,:,:]
        return inputs/255.0, labels


def testing(model_name, test_file):
    print('Testing')
    test_patches = PreparePatches(test_file)
    t_dataloader = DataLoader(test_patches, batch_size=32, shuffle=True, num_workers=4)
    model = Method(num_blocks=4, block_size=8, kernel_size=3,  depth=10)
    model.load_state_dict(torch.load(model_name))
    model.eval().cuda()
    loss_fn = nn.MSELoss()
    running_loss = 0.0
    psnr_sum=0.0
    counter = 0
    
    for data in tqdm(t_dataloader):
        counter += 1
        inputs, labels = data
        #interpolate to decide about the size of labels acc. to the patch size - 15/7/'22
        #labels = F.interpolate(labels,size=[64,64])
        inputs1,labels1 = inputs.cuda(), labels.cuda()
        #inputs1, labels1 = torch.unsqueeze(inputs1, 1), torch.unsqueeze(labels1, 1)
        #print("size of inputs1 and labels1:", inputs1.size())
        #optimizer.zero_grad()
        outputs1 = model(inputs1)[:, :31, :, :]  #changes
        
        # print('outputs in tr',outputs.shape)
        
        
        loss = loss_fn(outputs1, labels1) #changes
        
        running_loss += loss.item()
        
        outputs2 = outputs1.cpu().detach()
        labels2 = labels1.cpu().detach()
        # neeche wali line bhi thodi change kari hai
        psnr = batch_PSNR(outputs2,labels2)
        psnr_sum += psnr.item()
    
    psnr_sum = psnr_sum/counter
    print('av psnr',psnr_sum)
    
    epoch_loss = running_loss / counter
    print(f"Train Loss : {epoch_loss:.8f}")
 

#use this to calculate psnr(PSNR with data-range) -15/7/'22
def batch_PSNR(im_true, im_fake, data_range=255):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    Ifake = im_fake.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    mse = nn.MSELoss(reduction='none')
    err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C*H*W)
    psnr = 10. * torch.log((data_range**2)/err) / np.log(10.)
    return torch.mean(psnr)


if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print("3 args!!!")
        sys.exit(0)
    testing(sys.argv[1], sys.argv[2])
