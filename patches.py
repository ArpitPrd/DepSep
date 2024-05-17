import os
import h5py
import numpy as np
import cv2
from scipy.interpolate import interp1d
from shutil import copyfile
import matplotlib.pyplot as plt
import scipy.io

#Im2patch creates patches, use the def of process data below, instead of importing -15/7/'22
def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)

# # # # use this def when creating train patches (comment and un-comment accordingly)
def process_data(index, key, patch_size, stride, h5f, hyper_name, rgb_name, mode):   #hyper_name instead of rgb_name
    #hyper data as mat
    mathy =  h5py.File(hyper_name,'r')
    # mathy = scipy.io.loadmat(hyper_name)
    hyper = np.float32(np.array(mathy['rad']))
    print('before trans',hyper.shape)
    #for bgu,cave
    hyper = np.transpose(hyper, [2,0,1])
    print('AFTER trans',hyper.shape)
    hyper = normalize(hyper, max_val=255., min_val=0.)  #normalize rgb in harv
    print('hypershape',hyper.shape)
    #use normalize here instead of normalizing in matlab(always check in matlab for normalisation(do u even need it)) - 15/7/'22
    #for bgu
    hyper = normalize(hyper, max_val=4095., min_val=0.)
    #for cave
    #hyper = normalize(hyper, max_val=65535., min_val=0.)

#NO NORMALISATION FOR HARV.
        
    
    #rgb without interpolation 
    rgb =  cv2.imread(rgb_name)
    # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    print('before transpose',rgb.shape)    
    rgb = np.transpose(rgb, [2,0,1])
    print('trans-rgbshape',rgb.shape)
    rgb = normalize(np.float32(rgb), max_val=255., min_val=0.)  #normalize rgb in harv
  
    # rgb插值
    real_rgb = rgb # 3x64x64
    #print(real_rgb.shape)
    zeros = np.ones([28, real_rgb.shape[1], real_rgb.shape[2]], dtype=np.float32)
    real_rgb = np.append(real_rgb, zeros, axis=0)
    x1 = np.linspace(0, 1, num=3)
    x2 = np.linspace(0, 1, num=31)
    for m in range(real_rgb.shape[1]):
        for j in range(real_rgb.shape[2]):
            #print('m',m)
            f = interp1d(x1, real_rgb[0:3,m,j])
            real_rgb[:,m,j] = f(x2)
    print(real_rgb.shape)
    
    # create patches
    patches_hyper = Im2Patch(hyper, win=patch_size, stride=stride)
    patches_rgb = Im2Patch(real_rgb, win=patch_size, stride=stride)
    print('patches rgb',patches_rgb.shape)
    print('patches hyper',patches_hyper.shape)
    # add data
    for j in range(patches_hyper.shape[3]):
        print("generate training sample {}".format(index))
        sub_hyper = patches_hyper[:,:,:,j]
#         plt.imshow(sub_hyper[1,:,:])
#         plt.show()
        sub_rgb = patches_rgb[:,:,:,j]
        data = np.concatenate(( sub_rgb, sub_hyper), 0)
        h5f.create_dataset(str(index), data=data)
        index += 1
    return index

#### use this def when creating test patches (comment and un-comment accordingly)

# def process_data(index, key, patch_size, stride, h5f, hyper_name, rgb_name, mode):   #hyper_name instead of rgb_name
#     #hyper data as mat
#     mathy =  h5py.File(hyper_name,'r')
#     #mathy = scipy.io.loadmat(hyper_name)
#     hyper = np.float32(np.array(mathy['rad']))
#     #for bgu,cave
#     #hyper = np.transpose(hyper, [0,2,1])
#     #for harv
#     print('before trans',hyper.shape)
#     hyper = np.transpose(hyper, [0,2,1])
#     print('after trans',hyper.shape)
#     print('max(max)of hyper',hyper.max())
#     #use normalize here instead of normalizing in matlab(always check in matlab for normalisation(do u even need it)) - 15/7/'22
#     #for bgu
#     #hyper = normalize(hyper, max_val=4095., min_val=0.)
#     #for cave
#     #hyper = normalize(hyper, max_val=65535., min_val=0.)
#     print('hyper after norm and trans',hyper.shape)
#     plt.imshow(hyper[1,:,:])
#     plt.show()

#     # NO NORMALISATION FOR HARV& CAVE(because they have been normalised already in matlab using srf)
    
#     #rgb without interpolation 
#     rgb =  cv2.imread(rgb_name)
#     rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)    
#     rgb = np.transpose(rgb, [2,0,1])
#     print('max(max)of rgb',rgb.max())
#     rgb = normalize(np.float32(rgb), max_val=255., min_val=0.)
#     print('rgb after norm and trans',rgb.shape)
#     plt.imshow(rgb[1,:,:])
#     plt.show()

    
#     # rgb插值
#     real_rgb = rgb # 3x64x64
#     #print(real_rgb.shape)
#     zeros = np.ones([28, real_rgb.shape[1], real_rgb.shape[2]], dtype=np.float32)
#     real_rgb = np.append(real_rgb, zeros, axis=0)
#     x1 = np.linspace(0, 1, num=3)
#     x2 = np.linspace(0, 1, num=31)
#     for m in range(real_rgb.shape[1]):
#         for j in range(real_rgb.shape[2]):
#             #print('m',m)
#             f = interp1d(x1, real_rgb[0:3,m,j])
#             real_rgb[:,m,j] = f(x2)
#     print(real_rgb.shape)

# # Creating test data and storing them as key
    
#     data = np.concatenate((real_rgb,hyper), 0)
# #changed (create_dataset(key)) to (create_dataset(str(key)) -SH(19/05/'22')
#     h5f.create_dataset(str(key), data=data)
#     index += 1
#     return index

process_data(index = 1, key = 263, patch_size = 64, stride = 1, h5f = 'train.h5', hyper_name = 'data/BGU/train_spectral/BGU_HS_00263.mat', rgb_name = 'data/BGU/train_clean/BGU_HS_00263_clean.png', mode = 'train')