---
layout: post
comments: true
title: Study of Object Detection Techniques using MMDetection Toolkit
author: Ben Klingher, Erik Ren
date: 2023-01-29
---

<!--more-->

# Study of Object Detection Techniques using MMDetection Toolkit

## Introduction
Object detection is a crucial task in computer vision, which is particularly relevant to the field of autonomous driving. With that goal in mind as inspiration, this project focuses primarily on object detection for vehicles. We begin by implementing and comparing multiple recent object detection techniques using the MMDetection library, including two-stage detectors (e.g., R-CNN, Fast R-CNN, Faster R-CNN), one-stage detectors (e.g., YOLO, SSD), and anchor-free detectors (e.g., FCOS, CenterNet). We then apply the top performing models to vehicle datasets. Finally, we finetune/train a new model that performs object detection as well as classification on vehicles by make and model.

## Overview

In this project, we evaluate the performance of different object detection techniques using MMDetection on several general datasets, and then focus on evaluating and training the best models for vehicle related datasets. In particularly, we  start by exploring and comparing the effectiveness of of YOLO, SSD, Faster R-CNN and FCOS on the COCO and PASCAL-VOC datasets.

Then, we apply these models to the problem of vehicle detection, using the KITTI Benchmark Dataset and the Tsinghua-Tencent Traffic-Sign Dataset.

Finally, we plan to finetune one of the pretrained models to perform the new task of object detection combined with classficiation of the vehicles by make and model. For this purpose we will use the Stanford Car Dataset. We will also look at the datasets: CompCars, BoxCars and  MIO-TCD. We will experiment with multiple ways of building this model. Each specific pretrained object detection model will require different augmentation methods, so we will evaluate which model is best suited to this additional task. We intend to complete a model that will both identify vehicles in an image as well as label them with the correct make and model.

If we have time, we will also test our model on video using the Car Object Detection dataset on Kaggle.

## Datasets

* COCO
* PASCAL VOC
* KITTI Benchmark Dataset
* Tsinghua-Tencent Traffic-Sign Dataset
* Stanford Car Dataset 
* CompCars
* BoxCars
* MIO-TCD
* Car Object Detection https://www.kaggle.com/datasets/sshikamaru/car-object-detection

## Outcomes

* A comprehensive comparison of different object detection techniques.
* Insights into the strengths and limitations of different object detection methods and their application to vehicle recognition.
* An end-to-end object detection system for identifying and labeling vehicles by make and model.

## References

Vehicle Attribute Recognition by Appearance: Computer Vision Methods for Vehicle Type, Make and Model Classification
https://link.springer.com/article/10.1007/s11265-020-01567-6

Real-Time Vehicle Make and Model Recognition with the Residual SqueezeNet Architecture
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6427723/

Object Detection With Deep Learning: A Review
https://ieeexplore.ieee.org/abstract/document/8627998?casa_token=O2bJ9bs8fF8AAAAA:UitiBGhZBSdgAheBAPj9ZnGgW64oKa-bXSNibIaTk1oZAtDMGboHxcPq32fdaQTgN02tz0iZKA

YOLOX: Exceeding YOLO Series in 2021
https://arxiv.org/abs/2107.08430

ResNet strikes back: An improved training procedure in timm
https://arxiv.org/abs/2110.00476

FCOS: Fully Convolutional One-Stage Object Detection
https://arxiv.org/abs/1904.01355
