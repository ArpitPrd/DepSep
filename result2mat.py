import torch
import os
import numpy as np
import scipy.io as scio
from tqdm import tqdm
from train import batch_PSNR
from convmixer import AdapConvMixer
import dl

# import os
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# # from torch.utils.data.sampler import Sumax_split_size_mb
# from convmixer import AdapConvMixer # import from curr dir
# from torchvision.transforms import ToTensor, ToPILImage
# import torch.optim as optim
# from train import training, validate # import from curr dir
# from tqdm import tqdm
# import h5py # import from curr dir
# import matplotlib.pyplot as plt
# from train import batch_PSNR # import from curr dir
# from dl import PreparePatches, PreparevalPatches # import from curr dir

from matplotlib import pyplot as plt

def Result2Mat(model, val_dataloader, val_patches, best_model_path):
    print('Result to mat')
    # running_loss = 0.0
    # running_loss_2 = 0.0
    psnr_sum=0.0
    counter = 0
    all_time = 0
    
    # def get_testfile_list():
    #     #changed path to absolute path -  SH(19/05/'22)
    #     #path = '/home/user/Documents/Paper Aug 2022/data/Cave/'
    #     #path = '/home/user/Documents/Paper Aug 2022/data/Harvard/'
    #     path = './data/BGU/'
        
    #     test_names_file = open(os.path.join(path, 'test_names.txt'), mode='r')
        
    #     test_rgb_filename_list = []
    #     for line in test_names_file:
    #         line = line.split('/n')[0]
    #         hyper_rgb = line.split(' ')[0]
    #         test_rgb_filename_list.append(hyper_rgb)
        
    #     return test_rgb_filename_list
    
    # # test_rgb_filename_list = get_testfile_list()
    # test_rgb_filename_list = os.listdir('data/BGU/')
    # print('test_rgb_filename_list len : {}'.format(len(test_rgb_filename_list)))
    
    #this path belongs to the dir-path where the results are to be stored     - 26/07/'22 
    path = './data/BGU/'
    #path = '/home/user/Documents/Paper Aug 2022/data/Cave/'
    #path = '/home/user/Documents/Paper Aug 2022/data/Harvard/'
    if not os.path.exists(os.path.join(path, 'result')):
        os.mkdir(os.path.join(path, 'result'))

    with torch.no_grad():
        for i, data in tqdm(enumerate(val_dataloader), total=int(len(val_dataloader)/val_dataloader.batch_size)):
            
            counter += 1
            
            # file_name = test_rgb_filename_list[i].split('/')[-1].split('_')[-1])
            #add split('_')[-1]) for BGU , and split('-')[-1]) for other dataset accordingly - SH(26.05.22)
            # key = int(file_name.split('.')[0].split('_')[-1].split('-')[-1])
            # print(file_name, key)
            # inputs, labels = val_patches.get_data_by_key(str(test_rgb_filename_list[i][:-1])) # AP
            inputs, labels = data
            # inputs, labels = torch.unsqueeze(inputs, 0), torch.unsqueeze(labels, 0)
            inputs, labels = inputs, labels # cuda removed
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
            scio.savemat(os.path.join(path, 'result',val_patches.key(i)+'.mat'),{'rad':fake_hyper_mat , 'gt':gt})
            print(np.mean(fake_hyper_mat))
            print('sucessfully save fake hyper !!!')
            psnr = batch_PSNR(outputs,labels).item()
            print('test img [{}/{}], fake hyper shape : {}, psnr : {}'.format(i+1, counter, outputs.shape, psnr))
            psnr_sum += psnr
    print('average test psnr : {}'.format(psnr_sum/counter))
    return psnr_sum  

# val_patches = PreparevalPatches()
# print(len(val_patches))
# val_dataloader = DataLoader(val_patches)
# best_model_path = './data/BGU/checkpoints/' + 'august-0-3.439563179016113.pth'
# dim = 64
# depth = 10
# kernel_size = 5
# patch_size = 1
# model = AdapConvMixer(dim,depth,kernel_size,patch_size)
# model.load_state_dict(torch.load(best_model_path))
# Result2Mat(model = model, val_dataloader = val_dataloader, val_patches = val_patches, best_model_path = best_model_path)
