# Feed forward texture Synthesis in autonn #

Implements two feedforward texture synthesis methods in [autonn] (https://github.com/vlfeat/autonn) and [MatConvNet] (https://github.com/vlfeat/matconvnet). 
[1] [Spatial Adversarial Networks] (https://github.com/zalandoresearch/spatial_gan) 2016, zalando research
[2] [Texture Networks] (https://github.com/DmitryUlyanov/texture_nets) 2016 Ulyanov et al

The architecture in Spatial GANs was changed slightly, to including unpooling + conv instead of transpose conv and a least squares gan (LSGAN) loss instead 
of the DCGAN loss. Left result is [1] and right results in this repository. Both were run for 5k iterations and 4 pooling layers, with all other parameters
copied from [1]. 

![SGANs](https://i.imgur.com/THtpVjL.jpg)
![This repository](https://i.imgur.com/BAAzmGR.jpg)
