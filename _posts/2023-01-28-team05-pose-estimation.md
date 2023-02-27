---
layout: post
comments: true
title: Pose Estimation
author: Aristotle Henderson, John Houser
date: 2023-01-28
---

> This post is an investigation and review of pose estimation research and implementation.

<!--more-->

{: class="table-of-content"}

- TOC
    {:toc}

## 1. Introduction and Objective

### 1.1 Introduction

### 1.2 Objective

## 2. Pose Estimation With High-Resolution Learning

### 2.1 Original Downsampling Pipeline

### 2.2 Maintaining a High-Resolution Representation

## 3. Setup and Preparation

### 3.1 Environment

First up, environment setup and installation! Since Google Colaboratory has many of the Python packages we need to train and test this dataset, we'll assume that anyone replicating our work is using it. From here, the original model will be referred to as HRNet.

1. Open a new Juypter Notebook file in Google Colab and start a runtime with GPU acceleration.
2. Clone the HRNet code.

    ``` sh
    !git clone https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
    %cd deep-high-resolution-net.pytorch
    ```

3. Install the repository's required dependencies. Some dependencies in `requirements.txt` may not be available; in that case, use the most recent version. In case `pip` isn't working, try `conda`.

    ``` sh
    !pip install -r requirements.txt
    ```

4. Build the project's C++ libraries.

    ``` sh
    %cd lib
    !make
    ```

5. We'll be using the MPII Human Pose Dataset for training and testing, so you'll need to download them. First, create the directories we'll use to store this data.

    ``` sh
    %cd ..
    %mkdir data
    %mkdir data/mpii
    %mkdir data/mpii/annot
    %mkdir data/mpii/images
    ```

6. Then, you'll need to download the annotation files.

    ``` sh
    !gdown 1QeBJFAH8JDDH1Wl5uGhreFR5hpmXfmUE -O data/mpii/annot --folder
    ```

7. Finally, it's time to download the images. This'll take a while; it's almost 13GB of images, after all!

    ``` sh
    !wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
    !tar -C data/mpii/images -xvf mpii_human_pose_v1.tar.gz
    ```

### 3.2 Testing the Model

If you want to test HRNet with all of the given test data, the project has a script for that, but you'll need to either train the model or download weights first.

``` sh
%mkdir models
!gdown 14p2l1u19bLOm5p6esKyolYJyfE1_inLv -O models --folder
```

Here's the script, given for one of the HRNet models trained on the MPII dataset:

```sh
!python tools/test.py \
    --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_mpii/pose_hrnet_w32_256x256.pth
```

## 4. Proposed Ablations and Improvements

- We plan to train HRNet using a smaller dataset of lower resolution images to determine the impact of the high-resolution module on images where those pipelines won't carry as much data in the first place.
- We also plan to modify the HRNet model by adding more stages to the high-resolution pipeline.

## References

[1] Cao, Zhe, et al. "OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields." _IEEE Transactions on Pattern Analysis and Machine Intelligence_. 2019.

[2] Sun, Ke, et al. "Deep High-Resolution Representation Learning for Human Pose Estimation." _Conference on Computer Vision and Pattern Recognition_. 2019.

[3] Güler, Rıza Alp, et al. "DensePose: Dense Human Pose Estimation In The Wild." _Conference on Computer Vision and Pattern Recognition_. 2018.
