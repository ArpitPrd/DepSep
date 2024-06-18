# Reconstruction of Hyperspectral Images from RGB images using depth separable 3D Convolution
This is an initiative funded by the IRD unit IIT Delhi under the Summer Undergrad Research Award. We intent to produce and efficient and reliable reconstruction technique for HSI (**Hyperspectral Images**).

## Changes

1. The input from dl is changed from 31 to 32 by duplicating the 31st layer and appending it. Just for ease in grouping
2. convmixer changed completely.
- Method top level class.
- Input divided into 4 groups to make it a 4d tensor
- bandwise, pixelwise conv performed induvidually on each group
- later pointwise conv to the 3d tensor
- All of the above encapsulated in core_method
3. batch size reduced - we have less data and are using stochastic gradient descent. Let us make a vector out of all the d(Loss) / d(parameter) that every parameter possess after a forward pass for one batch, call that loss_grad vector. We make a backward pass by the average of losses over one batch. If the batch size is very large, the loss_grad vector may end up pointing in some other direction in the loss curve and head us in the wrong minima. And since we have less data we cannot hope to come back to local minima. Therefore smaller batch sizes make incremental changes to the loss_grad vector allowing come backs to local minima.
4. LR decreased, validation curve was going very hapazardly wrong.
5. Train1 = (Train 162, Val 21, Test 20 = Total 203), Train2 = (Train 43, Val 5, Test 5), Validation = (Test 5)
6. The testing is now performed induvidually on each from Train1, Train2, Val. However model every model performs the best on Val.
7. Look at v1_log.txt for information of training
8. test.py for testing is created
9. **Most Imp** - input to model must be divided by 255.0 for normalisation
10. An overal residual connection performs extremely well
11. Our model reached a peak of 50.8 psnr, Ma'ams model 51 psnr



