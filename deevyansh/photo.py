import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import torch
MODE =4
MODE=str(MODE)

def makephoto(data,name):
    band_index = 0
    data = data[0, band_index]

    # Normalize the data (if necessary) for proper image representation
    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Save the data as an image
    plt.imsave( f'{name}.png',data_normalized)
    plt.show()

# Load the .mat file
mat_file_path = f'result/{MODE}.mat'
mat_contents = scipy.io.loadmat(mat_file_path)
# print("MAT file keys1:", mat_contents.keys())
data = mat_contents['gt']
# print('Type1',data.shape)
makephoto(data,'output1')


import h5py
f=h5py.File('data/BGU/train.h5','r')
# print(list(f.keys()))
# print(f[str(MODE)].shape)
# print(type(f[str(MODE)]))
mat_contents=f[str(MODE)][()]
# print(type(mat_contents))
# print(mat_contents.shape)
data=mat_contents
data=torch.from_numpy(data)
data=torch.unsqueeze(data,0)
data=data.numpy()
print('Type2',data.shape)
makephoto(data,'output2')
