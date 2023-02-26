---
layout: post
comments: true
title: Multi View Stereo (MVS)
author: Hongzhe Du and Olivia Zhang
date: 2022-01-29
---

> This post provides an introduction to Multi View Stereo (MVS) and presents to deep learning based algorithms for MVS reconstruction. 

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Multi View Stereo (MVS)
Multi View Stereo (MVS) reconstructs a dense 3D geometry of an object or scene from calibrated 2D images taken from multiple angles. It is an important computer vision task as it is a pivotal step in robotics, augmented/virtual reality, automated navigation and more. Deep learning, with its success in computer vision tasks, has been increasingly used in solving 3D vision problems, including MVS. In this project, we are going to investigate two state-of-the-art MVS frameworks – TransMVSNet that uses a transformer-based deep neural network and CDS-MVSNet, which is a dynamic scale feature extraction network using normal curvature of the image surface.

## Related Work And Algorithms 
In this section, we are going to summarize the two MVS algorithms and the DTU MVS dataset this project focuses on. 

### TransMVSNet 
![TransMVSNet]({{ '/assets/images/06/TransMVSNet.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 1. TransMVSNet architecture* [1].

TransMVSNet is an end-to-end deep neural network model for MVS reconstruction. Fig. 1 shows the architecture of TransMVSNet. TransMVSNet first applies a Feature Pyramid Network (FPN) to obtain three deep image features with different resolutions. Then, the scope of these extracted features are adaptively adjusted through an Adaptive Receptive Field Module (ARF) that is implemented by deformable convolution with learnable offsets for sampling.  The features adjusted to the same resolutions are then fed to the FeatureMatching Transformer (FMT). The FMT first performs positional encoding to these features and flattens them. Then, the flattened feature map is fed to a sequence of attention blocks. In each block, all features first compute an intr-attention with shared weights, and then each reference feature is updated with a unidirectional inter-attention information from the source feature. The feature maps processed by FMT then go through a correlation volume and  3D CNNs for regularization to obtain a regularized probability volume. Then, winner-take-all is used to determine the final prediction. 


### CDS-MVSnet
![CDS-MVSnet]({{ '/assets/images/06/CDS-MVSnet.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 2. CDS-MVSnet architecture* [3].

CDS-MVSNet is a novel MVS framework aiming at improving the quality of MVS reconstruction while decreasing computation time and memory consumption. The CDS-MVSNet is composed of multiple CDSConv layers. Each CDSConv layer learns dynamic scale features guided by the normal curvature of the image surface. This operation is implemented by approximating surface normal curvatures in several candidate scales and choosing a proper scale via a classification network. These layers enable the CDS-MVSNet to estimate the optimal pixel’s scale to learn features adaptively with respect to structures, textures, and epipolar constraints.

### DTU MVS
The DTU MVS dataset is a publicly available dataset used for evaluating MVS algorithms. It was created by the Technical University of Denmark (DTU) and contains a variety of scenes, each captured from multiple viewpoints. The dataset includes 59 scenes that contain 59 camera positions and 21 that contain 64 camera positions. The scenes vary in geometric complexity, texture, and specularity. Each image is 1200x1600 pixels in 8-bit RGB color. 

![DTUMVS]({{ '/assets/images/06/DTUMVS.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 3. Examples of reference point clouds in DTUMVS dataset. The subset includes scenes of various texture, geometry, and reflectance* [1].

The dataset provides reference data (ground truth depth) by measuring a reference scan from each camera position and combining them. The DTU MVS dataset is widely used in the computer vision community as a benchmark for evaluating and comparing different multi-view stereo algorithms. We are going to use this dataset to reproduce and compare the results of the two algorithms that this project focuses on. 

## Codebase
The official implementation of TransMVSNet is available at [this repository](https://github.com/megvii-research/TransMVSNet). Source code of CDS-MVSNet is available at [this repository](https://github.com/TruongKhang/cds-mvsnet).

## Reference
[1] Aanæs, H., Jensen, R.R., Vogiatzis, G. et al. Large-Scale Data for Multiple-View Stereopsis. *Int J Comput Vis* 120, 153–168 (2016). 

[2] Ding, Yikang, et al. "Transmvsnet: Global context-aware multi-view stereo network with transformers." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2022.

[3] Giang, Khang Truong, Soohwan Song, and Sungho Jo. "Curvature-guided dynamic scale networks for multi-view stereo." *arXiv preprint arXiv:2112.05999* (2021).


