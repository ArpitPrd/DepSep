# Reconstruction of Hyperspectral Images from RGB images using depth separable 3D Convolution
This is an initiative funded by the IRD unit IIT Delhi under the Summer Undergrad Research Award. We intent to produce and efficient and reliable reconstruction technique for HSI (**Hyperspectral Images**).

## Vision

We wish to be able to prove that for depth separation in reconstruction of HSI (further in general case as well), is equally or better robust than the standard convolution robust. Also the shift from 2D conv in SOTA to 3D conv with depth separation in another level might reducing model params.

## Dataset

**Dataset** - https://csciitd-my.sharepoint.com/:f:/g/personal/ee3221883_iitd_ac_in/EsJYLZQrBTdFkqRUcLB-I-UB5CchpiaUJ0j1pPbIA9iMeA?e=XTV7sX

## Points to be addressed 

- Full fledged conv3d produced 54 psnr in 10 epochs with just 50 images
- With normal conv2d this is the case: Intial Layers did not reconstruct properly but near the end reconstruction seemed ok. Reason it the losses are averaged over the entire spectrum. This can possibly be addressed by weighing the losses differently. Basically not giving all of them equal weightage. The ones where loss is heavy give more weightage rest give less.
- With conv2d we achieved 48 psnr possibly because of less images and some of them were not being reconstructed properly
- Images had been seen


## Progress Report

### Week 1 & 2

  - Dev - Changed all convolution to 3d adjusted batches and sizes of al inputs
  - Ar - Generated data and ran main.py with conv2d setting
  - stride = patch_size (assumed) = 64
  - Ar - not included depth in the convmixer. temp training only with depth = 1
  - point wise convolution made naturally

### Week 3

  - Ar - patches.py corrected and transpose removed
  - Ar - depsep3d.py created - bandwise conv, pixelwise conv, pointwise conv
  - psnr value = ~48 with just 50 train images! When more images added it will reach 51 according to the paper
  - Trying the novel depsep
  - Dev - Finding errors in the code
  - Dev - Analyzing the params of the different models

### Week 4
  - Run Depsep3d model and check the losses

# Proposal 
The following is taken from the SURA proporsal 
## OBJECTIVE and MOTIVATION

- In-feasibility of Physical Measurement of HSI
- Reduction of Compute Requirements
- Increase Amount of reliable HSI images

## PROPOSED MODELS

- Usage of 3D Convolution instead of 2D Convolution
- Using Depth Separation in 3D Convolution
- Novel Depth Separation in Channel Wise Convolution

## Brief Explanation of Pipeline

RGB image is first interpolated to the number of spectra required at output. This is then fed into a convolutions layer that increases the amount of depth in the data. The data is then grouped uniformly which are then used as a 4D input to a 3D convolution layer which is depth-separated to band-wise, pixel-wise and point-wise. The output of this convolution is fed into another layer of convolution to restore the original spectral size of the input. The initially interpolated input is added to this to avoid vanishing gradients. The sum is the reconstructed HSI.

## Result Analysis Post Model Construction

- Estimate the accuracy of model after reduction of parameters
- Estimate the efficiency of depth separation for reconstruction
- Explore the generalisation of depth seperation in 3D Convolution
- Deduce Advantages and Disadvantages


