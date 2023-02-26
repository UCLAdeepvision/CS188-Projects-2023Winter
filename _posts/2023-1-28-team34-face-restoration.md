---
layout: post
comments: true
title: GAN Network Application towards Face Restoration
author: Michael Ryu
date: 2023-1-29
---


> Analyzing the Application of Generative Adversarial Networks For Face Restoration


<!--more-->
{: class="table-of-content"}

{:toc}

## Main Content
My project is to study the usage of Generative Adversarial Networks, and in particular the style-based GAN architecture, in order to generate photorealistic images with some noise involved, that allows it to be utilized in the application of face restoration. The models we will analyze will have been trained particularly on the human face, allowing it to enhance the image quality of real-world inputs. 

## Three Most Relevant Research Papers

1. This [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Towards_Real-World_Blind_Face_Restoration_With_Generative_Facial_Prior_CVPR_2021_paper.pdf) talks about utilizing a pre-trained GAN along with "rich and diverse priors" to be restore low-quality inputs.(1)

2. This [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_GAN_Prior_Embedded_Network_for_Blind_Face_Restoration_in_the_CVPR_2021_paper.pdf) explains how their GAN prior network is capable of being embedded into a U-Shaped DNN, and how their model is capable of restoring images that have degraded over time.(2)

3. This [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.pdf) talks about the redesigns of the original StyleGAN in order to improve upon its generative image modeling capabilities.

### Most Relevant Code
This [github repository](https://github.com/lucidrains/stylegan2-pytorch) implements a simple Pytorch implementation of StyleGan2, which is the style-based GAN Architecture.


## Reference
[1] MWang, Xintao, et al. "Towards real-world blind face restoration with generative facial prior." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

[2] Yang, Tao, et al. "Gan prior embedded network for blind face restoration in the wild." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

[3] Karras, Tero, et al. "Analyzing and improving the image quality of stylegan." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

---
