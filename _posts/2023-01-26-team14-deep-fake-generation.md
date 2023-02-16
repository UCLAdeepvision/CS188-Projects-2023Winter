---
layout: post
comments: true
title: Deep Fake Generation
author: Sarah Mauricio and Andres Cruz
date: 2023-01-26
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* Introduction
* What is Deepfake
    * Example: Image to Image Translation
    * Example: Image Animation
* Cycle-Consistent Adversarial Networks
    * Motivation
    * Architecture 
    * Architecture Blocks and Code Implementation
    * Results
* Conditional Adversarial Networks
    * Motivation
    * Architecture 
    * Architecture Blocks and Code Implementation
    * Results
* Demo
* References
{:toc}

## Introduction

We will be working on deep fake generation.

## What is Deepfake

### Example: Image-To-Image Translation

### Example: Image Animation

## Cycle-Consistent Adversarial Networks

### Motivation

### Architecture

### Architecture Blocks and Code Implementation

### Results

## Conditional Adversarial Networks

### Motivation

### Architecture

### Architecture Blocks and Code Implementation

### Results

## Demo

## References

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

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

---
