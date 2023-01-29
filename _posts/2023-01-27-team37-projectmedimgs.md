---
layout: post
comments: true
title: Medical Imaging
author: Team 37
date: 2022-01-27
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Main Content
Medical Imaging


Medical Image Segmentation refers to  the process of taking 2D or 3D image data and dividing  it into regions of interest. An example might be to take a Computed Tomography (CT) scan and isolate a particular anomaly within a patient. Doing medical image segmentation by hand can be a cumbersome task. Thus, the goal of this project is to automate this task with deep learning, with accuracy close to human standards.

Relevant Links:

[A collection of recent image segmentation methods, categorized by regions of the human body](https://github.com/JunMa11/SOTA-MedSeg)

[A student project from last year in this class. They have specifically studied PDV-Net and ResUNet++.](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2022/01/27/team07-medical-image-segmentation.html)

[A review paper discussing the various methods in deep learning used for medical imaging.](https://link.springer.com/article/10.1007/s12194-017-0406-5#Sec12)

[A medical imaging toolkit for deep learning.](https://github.com/fepegar/torchio/)

[A paper that focuses on data preparation of medical imaging data for use in machine learning.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7104701/)

[Meta analysis of diagnostic accuracy of deep learning methods in medical imaging.](https://www.nature.com/articles/s41746-021-00438-z)




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
