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
Pose estimation in humans is the process of locating key points in the human body including shoulders, knees, etc. Our project focused on 2D pose estimation which is concerned with identifying these keypoints in pictures of individuals. Specifically, our project was based on a paper that identifies the poses of individuals and not multi-person pose estimation. It has many applications including recognizing humans, their actions, animation, and more. One commonly cited example application is inferring the current action of the target. For example, by analyzing the pose of a human, we can determine if they are walking, running, or displaying another common action. The emergence of deep learning has greatly improved the capabilities of pose estimation. 

### 1.2 Objective
In this project we use High-Resolution Net (HRNet) to be able to estimate poses of subjects. We employ the Max Planck Institut Informatik (MPII) dataset in our implementation. This dataset includes pictures of individuals in various scenarios with annotations including keypoints of their pose. This data is crucial for our model's predictions and will be used to evaluate its performance.

The objective of this paper is to accurately predict the key points of a pose. We hope that our model is capable of making predictions about the locations of a subject's ankle, wrist, and other noteworthy points. Model performance is measured by comparing our predictions with the annotations provided by the MPII dataset.

Other goals include being able to discern the features that allow a model to accurately predict poses. This project will hopefully expose useful characteristics that can be applied to other applications of pose estimation. We realize that many of our results are limited by the diversity of our dataset. While the MPII dataset only includes 25K images, we hope that this is enough to achieve reasonable performance. It is undeniable that access to more data would certaintly increase the effectiveness of our model, but we limit our scope to this single dataset for practical reasons. The dataset may be unable to capture all the possible scenarios that our model may encounter in real situations, but our model should be able to identify the majority of images given to it in theory.

## 2. Pose Estimation With High-Resolution Learning

### 2.1 Original Downsampling Pipeline
The paper "Deep High-Resolution Representation Learning for Human Pose Estimation" serves as the backbone for our project. It introduces the novel architecture, HRNet, which utilizes a downsampling pipeline. The high to low process generates low-resolution and high-level representations. This idea allows the model to increase its performance while retaining many of the benefits of a model that maintains the same resolution as the input. The model also has its symmetric counterpart for restoring the representations back to the high resolution that the input possesses. The low to high process is targeted at producing high resolution representations. The paper uses a few bilinear upsampling or transpose convolution layers to restore this resolution.

<br>
![High to low ]({{ '/assets/images/HendersonHouser/Screenshot 2023-02-26 161702.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 1. Pose Estimation using the Downsampling Pipeline* [2].

### 2.2 Maintaining a High-Resolution Representation
The HRNet maintains the high-resolution representation by utilizing multi-scale fusion. The multi-resolution images are fed into the model and combined with the low to high process through the use of skip connections. Their implementation repeats multi-scale fusion which allows the model to retain simpler features despite the growth in its complexity. Additionally, the model is implemented with intermediate supervision. It helps the deep network train and improve the heatmap estimation quality.

The network connects high-to-low subnetworks in parallel and maintains high resolution representations with a spacially precise heatmap estimation. Most existing methods separate the low to high upsampling process and aggregate low and high level representations. The paper's approach does not use an intermediate heatmap supervision, but is superior in keypoint detection accuracy and efficient in computation complexity and parameters. The paper uses multi-resolution subnetworks that gradually adds high to low resolution subnetworks one by one, form new stages, and connect the multi-resolution subnetworks in parallel.

## 3. Setup and Preparation

### 3.1 Environment

### 3.2 Testing the Model

## 4. Proposed Ablations and Improvements

## References

[1] Cao, Zhe, et al. "OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields." _IEEE Transactions on Pattern Analysis and Machine Intelligence_. 2019.

[2] Sun, Ke, et al. "Deep High-Resolution Representation Learning for Human Pose Estimation." _Conference on Computer Vision and Pattern Recognition_. 2019.

[3] Güler, Rıza Alp, et al. "DensePose: Dense Human Pose Estimation In The Wild." _Conference on Computer Vision and Pattern Recognition_. 2018.
