import torch
import os
import numpy as np
import scipy.io as scio
from tqdm import tqdm
from train import batch_PSNR

from matplotlib import pyplot as plt

def Result2Mat(model, val_dataloader, val_patches):
    
     print('Result to mat')
    #model.train()
    # running_loss = 0.0
    # running_loss_2 = 0.0
     psnr_sum=0.0
     counter = 0
     all_time = 0
    
     def get_testfile_list():
          #changed path to absolute path -  SH(19/05/'22)
          #path = '/home/user/Documents/Paper Aug 2022/data/Cave/'
          #path = '/home/user/Documents/Paper Aug 2022/data/Harvard/'
          path = '/home/user/Documents/Paper Aug 2022/data/BGU/'

          test_names_file = open(os.path.join(path, 'test_names.txt'), mode='r')

          test_rgb_filename_list = []
          for line in test_names_file:
               line = line.split('/n')[0]
               hyper_rgb = line.split(' ')[0]
               test_rgb_filename_list.append(hyper_rgb)

          return test_rgb_filename_list
    
     test_rgb_filename_list = get_testfile_list()
     print('test_rgb_filename_list len : {}'.format(len(test_rgb_filename_list)))

     #this path belongs to the dir-path where the results are to be stored     - 26/07/'22 
     path = '/home/user/Documents/Paper Aug 2022/data/BGU/'
     #path = '/home/user/Documents/Paper Aug 2022/data/Cave/'
     #path = '/home/user/Documents/Paper Aug 2022/data/Harvard/'
     if not os.path.exists(os.path.join(path, 'result')):
          os.mkdir(os.path.join(path, 'result'))

     with torch.no_grad():
          for i, data in tqdm(enumerate(val_dataloader), total=int(len(val_dataloader)/val_dataloader.batch_size)):
                    counter += 1
                    file_name = test_rgb_filename_list[i].split('/')[-1]
     #add split('_')[-1]) for BGU , and split('-')[-1]) for other dataset accordingly - SH(26.05.22)
                    key = int(file_name.split('.')[0].split('_')[-1].split('-')[-1])
                    print(file_name, key)
                    inputs, labels = val_patches.get_data_by_key(str(key))
                    inputs, labels = torch.unsqueeze(inputs, 0), torch.unsqueeze(labels, 0)
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    fake_hyper_mat = outputs[0,:,:,:].cpu().numpy()
                    #print('max(max)of fh',fake_hyper_mat.max())
                    # print('fake_hyper_mat',fake_hyper_mat[15,:,:])
                    fake_hyper_plot = fake_hyper_mat[15,:,:]
                    gt = labels[0,:,:,:].cpu().numpy()
                    #print('max(max)of gt',gt.max())
                    # plt.imshow(gt[15,:,:])
                    # plt.show()
                    # plt.imshow(fake_hyper_plot)
                    # plt.show()
     # save output and GT to results dir  - 26/07/'22
                    scio.savemat(os.path.join(path, 'result',test_rgb_filename_list[i].split('/')[-1].split('.')[0]+'.mat'),{'rad':fake_hyper_mat , 'gt':gt})
                    print(np.mean(fake_hyper_mat))
                    print('sucessfully save fake hyper !!!')
                    psnr = batch_PSNR(outputs,labels).item()
                    print('test img [{}/{}], fake hyper shape : {}, psnr : {}'.format(i+1, counter, outputs.shape, psnr))
                    psnr_sum += psnr
          print('average test psnr : {}'.format(psnr_sum/counter))
          return psnr_sum  


