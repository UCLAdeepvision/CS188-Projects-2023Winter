---
layout: post
comments: true
title: Deep Fake Generation
author: Sarah Mauricio and Andres Cruz
date: 2023-01-26
---

## Abstract

> The use of deep learning methods in deep fake generation has contributed to the rise of fake of fake images which has some very serious ethical dilemmas. We will look at two different ways to generate deepfake pictures and videos, and will then focus in on Image-to-Image Translation. __ and __ are two different models we will by studying to create deep fake images using Image-to-Image Translation.
<!--more-->
## Table of Contents

* [Introduction](#intro)
* [What is Deepfake](#deepfake)
    * [Example: Image-to-Image Translation](#i2i)
    * [Example: Image Animation](#ia)
* [What is a Generative Adversarial Network (GAN)](#GAN)
* [Cycle GAN](#cycleGAN)
    * [Motivation](#mot1)
    * [Architecture](#arch1)
    * [Architecture Blocks and Code Implementation](#archBlocks1)
    * [Results](#res1)
* [Star GAN](#starGAN)
    * [Motivation](#mot2)
    * [Architecture](#arch2)
    * [Architecture Blocks and Code Implementation](#archBlocks2)
    * [Results](#res2)
* [Demo](#demo)
* [References](#ref)

## Introduction <a name="intro"></a>

We will be working on deep fake generation.

## What is Deepfake <a name="deepfake"></a>

Deepfake is a term used to describe artificially constructed media that portrays an individual or individuals in a way that suits the creator. For example, creating image to image translations, image animations, audio reconstruction, and more. Deepfakes are created using deep neural networks architectures, such as Generative Adversarial Networks or Autoencoders.

### Example: Image-to-Image Translation <a name="i2i"></a>

Image to image translation is the process of extracting features from a source image and emulating those features in another image. An example would be Neural Style Transfer, where a source image is used to create an art style to transfer to another image.

![Style Transfer](/assets/images/team14/style_transfer.png)
* Fig X. Example of Neural Style Transfer, https://www.v7labs.com/blog/neural-style-transfer

### Example: Image Animation <a name="ia"></a>

Image Animation is the action of generating a video where the object from an image is animated using the action from a driving video. For example, if we had an image of a water bottle and a driving video of a ball flying across the screen, the output video would be a water bottle flying across the screen. Thus, it will create an animation based on a single image.

![GAN Flow](/assets/images/team14/pipeline.png)
* Fig X. Example flow of Image Animation

Once applying the model, we would see results similar to the following:

![Image Animation Output](/assets/images/team14/vox-teaser.gif)
* Figure X. Example output from Image Animation


## What is a Generative Adversarial Network (GAN) <a name="GAN"></a>

Generative Adversarial Network, or GAN, is the core frameworkd behind a lot of the DeepFake algorithms you may come across. It is an approach to generate a model for a dataset using deep learning priciples. Generative modeling automatically discovers and learns the patterns in the data so that the model can be used to generate new images that could have been a part of the original dataset. GANs train a generative model that consists of two sub-components: the generator models which is trained to generate new images and the discriminator model which tries to classify an image as real or fake. The generative models and the discriminator model are trained together in an adversarial way, meaning until the discrimnator model classifies images incorrectly about half of the time. This would mean that the generator model generates DeepFake images that could pass as being real.

![GAN Flow](/assets/images/team14/gan1.JPG)
* Fig X. Example of GAN Flow

Below we look into two different models using ideas from GAN.

## Cycle GAN <a name="cycleGAN"></a>

### Motivation <a name="mot1"></a>

### Architecture <a name="arch1"></a>
 
### Architecture Blocks and Code Implementation <a name="archBlocks1"></a>

### Results <a name="res1"></a>

"preserve key attributes between the input and the translated image by utilizing a cycle consistency loss. However, [...] only capable of learning the relations between two different domains at a time. Their approaches have limited scalability in handling multiple domains since different models should be trained for each pair of domains" (Choi 3).

## Star GAN <a name="starGAN"></a>

### Motivation <a name="mot2"></a>
StarGAN is a generative adversarial network that learns the mappings among multiple domains using only a single generator and a discriminator, training effectively from images of all domains (Choi 2). The topology could be represented as a star where multi-domains are connected, thus receiveing the name StarGAN. 

![StarGAN Results](/assets/images/team14/star1.JPG)
* Fig X. Example of multi-domain image-to-image translation on CelebA dataset using StarGAN

StarGAN consists of two modules, a discriminator and a generator. The discriminator learns to differentiate between real and fake images and begins to clssify the real images with its proper domain. The gnerator takes an image and a target domain label as input and generates a fake image with them. The target domain label is then spatially replicated and concatenated with the image given as input. The generator attempts to reconstruct the orginal image via the fake image when given the original domain label. Lastly, the generator tries to generate images that are almost identical to the real images and will be classified as being from the target domain by the discriminator.

![StarGAN Flow](/assets/images/team14/star2.JPG)
* Fig X. Example flow of StarGAN

### Architecture <a name="arch2"></a>

### Architecture Blocks and Code Implementation <a name="archBlocks2"></a>

### Results <a name="res2"></a>

## Demo <a name="demo"></a>

## References <a name="ref"></a>

[1] Goodfellow, Ian J., et al. Generative Adversarial Networks. 2014. DOI.org (Datacite), https://doi.org/10.48550/ARXIV.1406.2661.

https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py

[2] Tolosana, Ruben, et al. DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection. 2020. DOI.org (Datacite), https://doi.org/10.48550/ARXIV.2001.00179.

https://github.com/deepfakes/faceswap

[3] Zhang, Tao. “Deepfake Generation and Detection, a Survey.” Multimedia Tools and Applications, vol. 81, no. 5, Feb. 2022, pp. 6259–76. DOI.org (Crossref), https://doi.org/10.1007/s11042-021-11733-y.

[4] Brownlee, Jason. “A Gentle Introduction to Generative Adversarial Networks (Gans).” MachineLearningMastery.com, 19 July 2019, https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/. 

https://github.com/Deepfakes/



## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).
