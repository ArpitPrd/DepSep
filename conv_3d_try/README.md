# Reconstruction of Hyperspectral Images from RGB images using depth separable 3D Convolution
This is an initiative funded by the IRD unit IIT Delhi under the Summer Undergrad Research Award. We intent to produce and efficient and reliable reconstruction technique for HSI (**Hyperspectral Images**).

## Reason for small batch size

It is SGD - parameter update happens by avg loss. If there is a lot of data in a single batch then the updaation vector may enter a random direction leading to incorrect learning. And since we have lesser data the loss might get into a different trajectory.

