# Feed forward texture Synthesis in autonn #

Implements two feedforward texture synthesis methods in [autonn](https://github.com/vlfeat/autonn) and [MatConvNet](https://github.com/vlfeat/matconvnet)
 
[1] [Spatial Adversarial Networks](https://github.com/zalandoresearch/spatial_gan) 2016, Zalando research

[2] [Texture Networks](https://github.com/DmitryUlyanov/texture_nets) 2016 Ulyanov et al


There are a few nuances, I do not use normalized gradients for [2], which leads to slower convergence. Also, I only evaluate VGG-19 to relu3_1 for memory. For [1], I periodize the boundaries and use LSGAN, instead of DCGAN. The results seem more crisp than [1] for some textures.

Original

![Original](https://i.imgur.com/xp2Dnda.jpg)


SGAN

![SGAN](https://i.imgur.com/soN1q2j.jpg)


TextureNet V1

![TextureNets V1](https://i.imgur.com/X5IbfIc.jpg)


