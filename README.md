# Reconstruction of Hyperspectral Images from RGB images using depth separable 3D Convolution
This is an initiative funded by the IRD unit IIT Delhi under the Summer Undergrad Research Award. We intent to produce and efficient and reliable reconstruction technique for HSI (**Hyperspectral Images**).

**Dataset** - https://csciitd-my.sharepoint.com/:f:/g/personal/ee3221883_iitd_ac_in/EsJYLZQrBTdFkqRUcLB-I-UB5CchpiaUJ0j1pPbIA9iMeA?e=XTV7sX

## Progress Week 1 & 2

- Dev - Changed all convolution to 3d adjusted batches and sizes of al inputs
- Ar - Generated data and ran main.py with conv2d setting
- stride = patch_size (assumed) = 64
- Ar - not included depth in the convmixer. temp training only with depth = 1
- point wise convolution made naturally

## Progress Week 3

- Ar - patches.py corrected and transpose removed
- Ar - depsep3d.py created - bandwise conv, pixelwise conv, pointwise conv
