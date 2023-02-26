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

In 2019, a team from Facebook AI Research ([FAIR](https://ai.facebook.com/)) published a paper that defined a new field of computer vision called **Panoptic Image Segmentation** that combines detections of *stuff* and *things* $$^{[1]}$$. However, before we can understand what panoptic segmentation is, we must understand some background.

![Cat]({{'/assets\images\team-11\cat.png' | relative_url}}){: style="width: 400px; max-width: 100%;"}
*Fig \#. Example cat image* $$^{[2]}$$.

### Stuff & Things

Stuff
:   amorphous and uncountable regions of similar texture or material such as grass, sky, road defined by simply assigning a class label to each pixel in an image <sup>[\[1\]](https://openaccess.thecvf.com/content_CVPR_2019/html/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.html)</sup> [^1]

Things
:   Items in an image that could possess more than 1 countable instance defined by detecting each object and delineating it with a bounding box or segmentation mask [^1]

Although identifying stuff and things sound like similar problems, the deep learning models that perform the task vary substansially in datasets, details, and metrics[^1]

### Semantic Segmentation

Semantic segmentation is a task that indentifies stuff. **Description of how it works**

![Semantic Cat]({{'/assets\images\team-11\sem_cat.png' | relative_url}}){: style="width: 400px; max-width: 100%;"}
*Fig \#. Semantic Segmentation on the cat image* [1].

### Instance Segmentation

Description of how it works

![Instance Cat]({{'/assets\images\team-11\inst_cat.png' | relative_url}}){: style="width: 400px; max-width: 100%;"}
*Fig \#. Instance Segmentation on the cat image* [1].

### Panoptic Segmentation

The paper sets the groundwork for the panoptic image segmenation problem to reconcile the dichotomy between *stuff* and *things* by combining semantic and instance segmentation

![Panoptic Cat]({{'/assets\images\team-11\pan_cat.png' | relative_url}}){: style="width: 400px; max-width: 100%;"}
*Fig \#. Panoptic Segmentation on the cat image* [1].

Kirillov et al. defines a $$PS$$ *(Panoptic Score)* and the requirements for a model to be considered a "panoptic segmentation model."

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

Make a directory in the drive called `MMDet1` and clone the MMDetection github repository into it and install with pip.

```python
!mkdir -pv /content/drive/MyDrive/MMDet1
%cd /content/drive/MyDrive/MMaDet1

# Install mmdetection
!rm -rf mmdetection
!git clone https://github.com/open-mmlab/mmdetection.git
%cd mmdetection

!pip install -e .    
```

MMdetection is now downloaded on your Google Drive and installed in your Colab session. You should **never** have to reclone the github repo!

### COCO-2017 Download

Now we will download the COCO-2017 dataset with the annotations for panoptic segmentation. First, make sure you are in the mmdetection folder in your drive, then use the `download_dataset.py` script to download the base COCO-2017 dataset. Then you will need to download the panoptic annotations. Next, unzip all the newly downloaded files into the `data/coco` directory.

The following code block will take about 5 hours to run fully. If you do not wish to wait all that time and would rather unzip in bursts, you can comment out all but the unzip command that your want to execute and run the cell for each unzip command. This allows you to unzip the files in chunks of time instead of all at once.

> **Note:** <br>
    1. You will need a GPU runtime to run the `download_dataset.py` python script. <br>
    2. Sometimes the dataset will not fully unzip and not let you know. Make sure to rerun this block until there are not outputs of newly extracted filenames.

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
!unzip -u data/coco/annotations/panoptic_train2017.zip -d data/coco/annotations
!unzip -u data/coco/annotations/panoptic_val2017.zip -d data/coco/annotations
```

Finally, convert the standard COCO annotations to the panoptic annotations with the `gen_coco_panoptic_test_info.py` script.

```python
!python tools/misc/gen_coco_panoptic_test_info.py data/coco/annotations
```

You are now ready to get started with working with MMDetection in Colab. Please make sure that you install `mmcv`, `cocodataset/panopticapi`, enter the mmdetection directory and run `pip install -e .` with each new document.

## PanopticFPN with MMDetection

### Background

The Panoptic FPN was designed as a single-network baseline for the panoptic segmentation task. They do this by starting from Mask R-CNN, a popular isntance segmentation model, with a Feature Pyramid Network (FPN) backbone based on ResNet. In parallel, they create a minimal semantic segmentation branch using the same features of the FPN to generate a dense-pixel output [^3]. The author's goal is to maintain top of the line performance for segementation quality ($$SQ$$) and recognition quality ($$RQ$$) [^3]

![FPN]({{'/assets\images\team-11\fpn.png' | relative_url}}){: style="width: 400px; max-width: 100%;"}
*Fig \#. Instance Segmentation on the cat image* [1].

#### Feature Pyramid Network

The FPN consists of a botton up pathway and a top-down pathway. The bottom-up pathway consists of feature maps of several scales with a scaling step of 2. Each step corresponds to a residual block stage from Resnet and the output of each step is the output of the activation function of the residual block (except for the first stage since it is so large). The stages have strides {4, 8, 16, 32} in order to downsample the feature map.

![Top_Down]({{'/assets\images\team-11\top_down_fpn.png' | relative_url}}){: style="width: 400px; max-width: 100%;"}
*Fig \#. Instance Segmentation on the cat image* [1].

The top-down pathway starts from the deepest layer of the network and progressively upsamples it while adding in transformed versions of higher-resolution features from the bottom-up pathway. The higher stages of the top-down pathway are at a smaller resolution, but semantically stronger. The purpose of the top-down pathway is to use this information to make a spatially fine and semantically stronger feature map of the input. Finally, the output of each stage of the top-down pathway is the final output of the FPN (labeled predict in Fig. #).

#### Instance Segmentation Branch

Mask R-CNN is an extension on Faster R-CNN that adds an masking head branch to predict an binary mask for each bounding box prediction. Panoptic FPN uses the Mask R-CNN with the ResNet FPN as a backbone from [#] since it has been used as a foundation for all top entries in recent recognition challenges[3].

#### Semantic Segmentation Branch

The semantic segmentation branch also builds on the FPN in parallel with the instance segmentation branch. This semantic segmentation branch was designed to be as simple as possible and so it only upsamples each output of the FPN layers to 1/4th total size, add each together, and perform a 1x1 conv with a 4x bilinear upsampling. Each upsampling layer consists of a 3x3 convolution, group norm, ReLU, and 2x bilinear upsampling. It is important to note that in addition to each of the stuff class of the dataset, the branch can also output a 'other' class for pixels that do not belong to any classes. This avoids the branch predicting the pixels belong to no class as a incorrect class.

![Semantic_Diagram]({{'/assets\images\team-11\semantic_diagram.png' | relative_url}}){: style="width: 400px; max-width: 100%;"}
*Fig \#. Instance Segmentation on the cat image* [1].

### Setup

Before we do anything, let's make sure we have our Colab environment set up correctly. Mount you Google Drive, install `mmcv`, `cocodataset/panopticapi`, and install the mmdetection library in your Colab session.

```python
import os
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Install mmcv-full
!pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

# Install cocodataset/panopticapi
!pip install git+https://github.com/cocodataset/panopticapi.git

# Install mmdetection in colab environment
%cd /content/drive/MyDrive/MMDet1/mmdetection
!pip install -e .
```

Panoptic Evaluation Results:
[Implementation Link](https://colab.research.google.com/drive/11MitSydv7qZ_xQkcLO4X2azTuGORjrQf#scrollTo=L-9pCPGHIkdo&uniqifier=2)

| Panoptic 1x ResNet Coco | PQ     | SQ     | RQ     | categories |
| :--------------- | :---: | :---: | :---: | :---: |
| All    | 40.248 | 77.785 | 49.312 | 133        |
| Things | 47.752 | 80.925 | 57.475 | 80         |
| Stuff  | 28.922 | 73.046 | 36.991 | 53         |




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

## Evaluation 

## Summary

## Conclusion

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

[^2] [What is Panoptic Segmentation and why you should care.](https://medium.com/@danielmechea/what-is-panoptic-segmentation-and-why-you-should-care-7f6c953d2a6a)

[^3] [Panoptic Feature Pyramid Networks, Kirillov et al. 2019](https://arxiv.org/pdf/1901.02446.pdf)

[^4] [FPN Paper](https://arxiv.org/pdf/1612.03144.pdf)


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
