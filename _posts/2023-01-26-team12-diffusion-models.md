---
layout: post
comments: true
title: Generating Images with Diffusion Models
author: Dave Ho, Anthony Zhu
date: 2023-01-26
---


> This project explores the latest technology behind Image-generative AIs such as DALLE-2 and Imagen. Specifically we'll be going over the research and techniques behind Diffusion Models, and a toy implementation in Pytorch/COLAB.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Project Proposal (WEEK 3)
Team 12:
* Dave Ho
* Anthony Zhu

Project Topic: **Diffusion Models for Generating Images**

Papers:
* [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
    * [Github](https://github.com/hojonathanho/diffusion)
* [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
    * [Github](https://github.com/openai/guided-diffusion)
* [A Path to the Variational Diffusion Loss](https://blog.alexalemi.com/diffusion.html)
    * [COLAB](https://colab.research.google.com/github/google-research/vdm/blob/main/colab/SimpleDiffusionColab.ipynb)
* [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)


# IGNORE BELOW [IN CONSTRUCTION]

## Main Content
Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

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
