import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from convmixer import AdapConvMixer
import h5py


def norm(data):
	return (data - data.min()) / (data.max() - data.min())

def im_view(file_pth, model_pth):
	model = AdapConvMixer(dim = 64, depth=10, kernel_size=5, patch_size=1)
	model.load_state_dict(torch.load(model_pth))
	model.eval()
	
	im = h5py.File(file_pth)
	rgb = np.int16(im['1'][:31, :, :])
	spec = np.int16(im['1'][31:, : ,:] * 500.0)
	rgb = np.transpose(rgb, [2, 1, 0])
	spec = np.transpose(spec, [2, 1, 0]) 
	
	rgb_in = torch.unsqueeze(torch.from_numpy(im['1'][:31, :, :]).float(), 0)
	spec_out = model(rgb_in)
	spec_out = torch.squeeze(spec_out, 0)
	spec_out = np.transpose(spec_out.detach().numpy(), [2, 1, 0])
	
	frq = 0
	plt.imshow(rgb[:, :, frq])
	plt.show()
	plt.imshow(spec[:, :, frq])
	plt.show()
	plt.imshow(spec_out[:, :, frq])
	plt.show()
	
	print("rgb", rgb.shape)
	print(rgb)
	print("spec", spec.shape)
	print(spec)
	print("spec_out", spec_out.shape)
	print(spec_out)


file_pth = '../data/BGU/val.h5'

model_pth = '../data/BGU/checkpoints_2d_depsep/august-99-48.195741617111935.pth'

im_view(file_pth, model_pth)

