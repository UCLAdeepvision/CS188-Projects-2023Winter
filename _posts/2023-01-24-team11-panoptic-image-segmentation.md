---
layout: post
comments: true
title: Analysis of Panoptic Image Segmentation Performance   
author: Andrew Fantino and Nicholas Oosthuizen
date: 2023-01-24
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.

<!--more-->
{: class="table-of-content"}

* TOC
{:toc}

## Introduction

In 2019, a team from Facebook AI Research ([FAIR](https://ai.facebook.com/)) published a paper that defined a new field of computer vision called **Panoptic Image Segmentation** that combines detections of *stuff* and *things* [^1]. However, before we can understand what panoptic segmentation is, we must understand some background.

### Stuff & Things

Stuff
:   amorphous and uncountable regions of similar texture or material such as grass, sky, road defined by simply assigning a class label to each pixel in an image [^1]

Things
:   Items in an image that could possess more than 1 countable instance defined by detecting each object and delineating it with a bounding box or segmentation mask [^1]

Although identifying stuff and things sound like similar problems, the deep learning models that perform the task vary substansially in datasets, details, and metrics[^1]

### Semantic Segmentation

Semantic segmentation is a task that indentifies stuff. **Description of how it works**

.**Add example Image**

### Instance Segmentation

Description of how it works

### Panoptic Segmentation

The paper sets the groundwork for the panoptic image segmenation problem to reconcile the dichotomy between *stuff* and *things* by combining semantic and instance segmentation

Kirillov et al. defines a $PS$ *(Panoptic Score)* and the requirements for a model to be considered a "panoptic segmentation model."

$$
PS= \frac{ \sum_{ (p,g) \in TP} IoU(p,g) }{ |TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN| }
$$

.**describe what each term means in the equation**

Kirillov et al. defines the panoptic segmentation format algorithm to map each pixel to an semantic class and an optional instance class. **Continue explanation**

We will be evaluating multiple panoptic segmentation models **continue with what we are going to do** (COCO-2017 dataset)

## MMDetection Setup

We will be evaluating and modifying the panoptic segmentation models from the [MMDetection ModelZoo](https://github.com/open-mmlab/mmdetection#overview-of-benchmark-and-model-zoo) using Google Colab for development and a GCP for longer training. Therefore, we will need to install MMdetection and download the COCO-2017 dataset.

### Install MMDetection in Google Colab

Luckily, a student group from the Winter2022 quarter of CS188 did most of the hard work with setting up MMDetection for Google Colab. They downloaded MMdetection and the COCO-2017 dataset to their Google Drive. [MMDetecton Project](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2022/02/20/team09-MMDetection.html#2-mmdetection-setup-and-data-preparation)

Unfortunately, we can't just copy-and-paste the entire setup since there have been changes to the installation steps and versions in Pytorch, mmcv, and Pillow. In addition, we need to download the additional panoptic segmentation labels.

To begin, we create a new Colab notebook that we will (hopfully) run only once to install MMDetection and download the COCO-2017 dataset. In our case, we named it `setup_mmdet_and_download_COCO.ipynb`.

First, mount your drive, and install mmcv in our colab environment. There is no need to install a special version of Pytorch or Pillow in this version of Colab.

```python
import os
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# install mmcv-full
!pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
```

Make a directory in the drive called `MMDet1` and enter it. Clone the MMDetection github repository and install it with pip.

```python
!mkdir -pv /content/drive/MyDrive/MMDet1
%cd /content/drive/MyDrive/MMaDet1

# Install mmdetection
!rm -rf mmdetection
!git clone https://github.com/open-mmlab/mmdetection.git
%cd mmdetection

!pip install -e .    
```

MMdetection is now downloaded on your Google Drive and installed in your Colab session.

### COCO-2017 Download

Now we will download the COCO-2017 dataset with the annotations for panoptic segmentation. First, make sure you are in the mmdetection folder in your drive, then use the `download_dataset.py` script to download the base COCO-2017 dataset. Then you will need to download the panoptic annotations. Next, unzip all the newly downloaded files into the `data/coco` directory.

The following code block will take about 5 hours to run fully. If you do not wish to wait all that time and would rather unzip in bursts, you can comment out all but the unzip command that your want to execute and run the cell for each unzip command. This allows you to unzip the files in chunks of time instead of all at once.

**Note**: you will need a GPU runtime to run the `download_dataset.py` python script.

```python
# suppose data/coco/ does not exist
!mkdir -pv data/coco/

# download the coco2017 dataset
!python3 tools/misc/download_dataset.py --dataset-name coco2017 

# Adjust the dataset to include panoptic annotations
!wget -P data/coco/ http://images.cocodataset.org/annotations/image_info_test2017.zip
!wget -P data/coco/ http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip

# unzip them
!unzip -u "data/coco/annotations_trainval2017.zip" -d "data/coco/"
!unzip -u "data/coco/test2017.zip" -d "data/coco/"
!unzip -u "data/coco/train2017.zip" -d "data/coco/"
!unzip -u "data/coco/val2017.zip" -d "data/coco/"

!unzip -u data/coco/image_info_test2017.zip -d data/coco/
!unzip -u data/coco/panoptic_annotations_trainval2017.zip -d data/coco/
```

Finally, convert the standard COCO annotations to the panoptic annotations with the `gen_coco_panoptic_test_info.py` script.

```python
!python tools/misc/gen_coco_panoptic_test_info.py data/coco/annotations
```

You are now ready to get started with working with MMDetection in Colab. Please make sure that you install mmcv, enter the mmdetection directory and run `!pip install -e .` with each new document.

## PanopticFPN with MMDetection

### Background

Here is some background on panopticFPN network. What is a FPN? Why do we care about this model?

### Setup

Before we do anything, let's make sure we have our Colab environment set up correctly. Mount you Google Drive, install mmcv, and install the mmdetection library in your session.

```python
import os
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Install mmcv-full
!pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

# Install mmdetection in colab environment
%cd /content/drive/MyDrive/MMDet1/mmdetection
!pip install -e .
```

## Maskformer with MMDetection

### M.1 Background

### M.2 Setup

Before we do anything, let's make sure we have our Colab environment set up correctly. Mount you Google Drive, install mmcv, and install the mmdetection library in your session.

```python
import os
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Install mmcv-full
!pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

# Install mmdetection in colab environment
%cd /content/drive/MyDrive/MMDet1/mmdetection
!pip install -e .
```

## Jan. 29 Submission

### Topic: Panoptic Segmentation

Andrew Fantino and Nicholas Oosthuizen will explore the topic of panoptic segmentation. They will describe the concepts behind it and assess several different models, describing their architectures and comparing their performance.

### Relevant Papers

1. [Panoptic Segmentation, Kirillov et al.(2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.html)
    * This paper sets the groundwork for the panoptic image segmenation problem by strictly defining the problem. It defines a $PS$ *(Panoptic Score)* and the requirements for a model to be considered a "panoptic segmentation model."

    $$
    PS= \frac{ \sum_{ (p,g) \in TP} IoU(p,g) }{ |TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN| }
    $$

2. [Panoptic Feature Pyramid Networks, Kirillov et al. (2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Feature_Pyramid_Networks_CVPR_2019_paper.pdf)
    * **Github:** [panoptic_fpn](https://github.com/open-mmlab/mmdetection/tree/master/configs/panoptic_fpn)
    * This paper is one of the first implementations of panoptic image segmentation from facebook research's paper above. This paper attempts to merge an semantic segmentation model and an instanse segmentation model using as little of a transformer network as possible. It is meant as a baseline evaluation of a panoptic segmentation model.
3. [Per-Pixel Classification is Not All You Need for Semantic Segmentation, Cheng et al. (2021)](https://arxiv.org/pdf/2107.06278.pdf)
    * **Github:** [maskformer](https://github.com/open-mmlab/mmdetection/tree/master/configs/maskformer)
    * This is an example of a panoptic segmentation model that is designed from the ground up for maximizing $PS$ that was included in MMDection.
4. [Masked-attention Mask Transformer for Universal Image Segmentation, Cheng et al. (2022)](https://arxiv.org/pdf/2112.01527.pdf)
    * **Github:** [mask2former](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former)
    * This is a second version of the paper 3. It was also included in MMDetectoin

## References

[^1] [Panoptic Segmentation, Kirillov et al.(2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.html)

<!-- ## Main Content

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

```python
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

--- -->
