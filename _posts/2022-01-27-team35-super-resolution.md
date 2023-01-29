---
layout: post
comments: true
title: Single Image Super Resolution
author: Ethan Truong, Archisha Datta
date: 2022-01-28
---


> Below is our setup for part 1...


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Three Most-Relevent Research Papers
1. #### Learning a Single Convolutional Super-Resolution Network for Multiple Degradations
([Code](https://github.com/cszn/SRMD)) Uses CNNs to upscale images with non-standard degradation. This method is more scalable to data encountered in the real world. [1]

2. #### Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
([Code](https://github.com/tensorlayer/srgan)) Addresses the issue of retaining fine textural details during the upscaling process by using a loss function motivated by perceptual similarity rather than pixel similarity. [2]

3. #### Image Super-Resolution Using Deep Convolutional Networks
([Code](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)) Deep neural network approach to image super-resolution. [3]


## References
[1] Zhang, Kai, et al. ["Learning a Single Convolutional Super-Resolution Network for Multiple Degradations."](https://paperswithcode.com/paper/learning-a-single-convolutional-super) *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018.

[2] Ledig, Christian, et al. ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network."](https://paperswithcode.com/paper/photo-realistic-single-image-super-resolution) *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2017.

[3] Dong, Chao, et al. ["Image Super-Resolution Using Deep Convolutional Networks."](https://paperswithcode.com/paper/image-super-resolution-using-deep) *IEEE Transactions on Pattern Analysis and Machine Intelligence*. 2014.

---
