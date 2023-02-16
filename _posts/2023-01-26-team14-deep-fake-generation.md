---
layout: post
comments: true
title: Deep Fake Generation
author: Sarah Mauricio and Andres Cruz
date: 2023-01-26
---


> The use of deep learning methods in deep fake generation has contributed to the rise of fake of fake images which has some very serious ethical dilemmas. We will look at two different ways to generate deepfake pictures and videos, and will then focus in on Image-to-Image Translation. __ and __ are two different models we will by studying to create deep fake images using Image-to-Image Translation.
<!--more-->

* [Introduction](https://github.com/acruz0426/CS188-Projects-2023Winter/blob/main/_posts/2023-01-26-team14-deep-fake-generation.md#introduction-)
* [What is Deepfake] (#deepfake)
    * [Example: Image-to-Image Translation] (#i2i)
    * [Example: Image Animation] (#ia)
* [What is Generative Adversarial Networks (GAN)] (#GAN)
* [Cycle GAN] (#cycleGAN)
    * [Motivation] (#mot1)
    * [Architecture] (#arch1)
    * [Architecture] Blocks and Code Implementation (#archBlock1)
    * [Results] (#res1)
* [Star GAN] (#starGAN)
    * [Motivation] (#mot2)
    * [Architecture] (#arch2)
    * [Architecture Blocks and Code Implementation] (#archBlock2)
    * [Results] (#res2)
* [Demo] (#demo)
* [References] (#ref)

## Introduction <a name="intro"></a>

We will be working on deep fake generation.

## What is Deepfake <a name="deepfake"></a>

### Example: Image-to-Image Translation <a name="i2i"></a>

### Example: Image Animation <a name="ia"></a>

## What is Generative Adversarial Networks (GAN) <a name="GAN"></a>

## Cycle GAN <a name="cycleGAN"></a>

### Motivation <a name="mot1"></a>

### Architecture <a name="arch1"></a>
 
### Architecture Blocks and Code Implementation <a name="archBlocks1"></a>

### Results <a name="res1"></a>

## Star GAN <a name="starGAN"></a>

### Motivation <a name="mot2"></a>

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
