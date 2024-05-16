
## Making of the h5py file
import h5py
file=h5py.File("RGB_to_HSI.hdf5","w")
dataset=file.create_dataset("data",shape=(62,64,64),dtype='i')


for i in range (0,62):
    for j in range (0,64):
        for k in range (0,64):
            dataset[i,j,k]=i+j+k

file.close()

## Checking of the h5py file

import h5py
import numpy as np
import torch
file=h5py.File("RGB_to_HSI.hdf5","r")
dataset=file['data']
data=np.array(dataset)
data=torch.tensor(data)
print(data[0:31])
print(data)
