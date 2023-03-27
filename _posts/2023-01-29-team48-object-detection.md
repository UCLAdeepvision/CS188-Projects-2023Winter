---
layout: post
comments: true
title: Object Detection
author: Jay Jay Phoemphoolsinchai
date: 2022-01-29
---


# Use Case Object Detection
Object detection is an incredibly important fundamental idea; many advanced and sophisticated tasks are directly impacted by the performance of an object detection model. In this article, we take a close look at developments in the object detection field of computer vision that may or may not lend themselves more towards a particular use case.

- [Overview Video](#overview-video)
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Method](#method)
- [Results](#results)
- [Discussion](#discussion)

# Overview Video

# Abstract
To see if different characteristics of object detection models can lead to significantly differing performances for different classes. A variety of models were chosen, for a total of five. Models were fine-tuned and ran on a subset of the COCO 2017 validation dataset for particular classes. The results show that there may be a slight link, but due to the factor of confounding variables, not many conclusions can be drawn.

# Introduction 
The task being worked on is object detection, using the COCO 2017 dataset. The goal is to see if particular characteristics of object detection models lend themselves more towards a particular type of object. The hypothesis is that there will be no link, as images are simply just a bunch of pixels when fed into an object detection model. However, there is the possibility that particular objects will tend to have similar pixel patterns, and if there is a way that certain components allow object detection models to perform better on particular patterns compared to other patterns, then this would be interesting and potentially valuable information.

If any conclusive results can be drawn from these experiments, it will show that there is a possibility that certain components of object detection models can be harnessed to their full potential in certain areas of application. If particular components can be shown to be significantly better than others when it comes to detecting certain things, it will help direct future models towards those components in the name of optimal progress.

# Method
Filtering the COCO 2017 dataset comes first, and is done with a relatively simple Python script that combs through the validation and train `json` files, filtering out all but the desired category, mapping the old category ids to new ones such that they begin at 1, updating the category ids in the annotations, filtering out all annotations that don't correspond to images of the desired category, and finally filtering out all images that don't belong to the desired category. In this instance, the chosen categories were `cow`, `banana`, and `backpack`. Then, we can use the newly generated `json` files to download all the images for use later.

Once we have the `json` files and all the images downloaded, we can begin model evaluation using Facebook/Meta AI Research's Detectron2 library. We will be testing five Faster R-CNN models:
1. ResNet-50 Conv-4
2. ResNet-50 Dilated-Conv-5
3. ResNet-50 Feature Pyramid Network
4. ResNet-101 Feature Pyramid Network
5. ResNeXt-101-32x8d Feature Pyramid Network

Each model will be fine-tuned on each new dataset (`cow`, `banana`, and `backpack`) and then evaluate the performance using the AP metric. Each metric will be compared across classes, including the baseline (entire COCO dataset) and also across models.

# Results
| Model            | baseline | cow | banana | backpack |
| ---------------- | -------- | --- | ------ | -------- |
| ResNet-50 Conv-4 | 35.7 | 51.745 | 17.241 | 11.790 |
| ResNet-50 Dilated-Conv-5 | 37.3 | 49.004 | 15.596 | 15.130 |
| ResNet-50 Feature Pyramid Network | 37.9 | 52.226 | 15.569 | 13.650 |
| ResNet-101 Feature Pyramid Network | 42 | 52.499 | 18.906 | 18.667 |
| ResNeXt-101-32x8d Feature Pyramid Network | 43 | 51.304 | 15.449 | 21.694 |

> Code can be found [here](https://drive.google.com/drive/folders/1Dpge3DJ8stn-zr1g6vb3YoDvLnlyfFqQ?usp=sharing).

# Discussion
The models are listed in ascending order of baseline AP. However, we can see that the AP for the various classes are not in ascending order. We can see some interesting observations here. For instance, based on the three different ResNet-50 models, it may show that Dilated-Conv-5 performs the worst for the `cow` class yet the best for the `backpack` class and that Conv-4 performs the best for the `banana` class, even though Feature Pyramid Network performs the best overall when it comes to the entire COCO dataset. In these subsets however, it only performs the best in the `cow` class, showing that, for example, Conv-4 may be better than Feature Pyramid Network when it comes to detecting the `banana` class, even if Feature Pyramid Network is better overall. This observation may show that certain components of an object detection model can indeed be more suited for some classes than others.

Ultimately, no real conclusions can be made from this data due to confounding variables. There are a number of improvements that could be applied here in the future. For one, the results may have been influenced by the relatively low dataset size. As we know, low dataset size can have big impacts on the performance of models. While the entire COCO dataset is quite huge and more than sufficient, choosing only one class narrows the dataset to an extreme amount, perhaps too much. This could be rectified by choosing a supercategory to filter on instead of just one category; this was attempted during this experiment, but ultimately Colab limitations prevented this vein of continuation (drive timeout). Also, if we could somehow make COCO size-equivalent datasets for specific categories, this would be very useful. On the other hand, going away from COCO may help. Since the models were already trained on COCO, fine-tuning using COCO is not ideal. For example, if we had a giant dataset of only `banana` related images, this may produce clearer results for the `banana` class. 

Another improvement could be using more models. Only fifteen models were fine-tuned for the interest of time since this is only an exploratory experiment, but using more would be much better for seeing patterns. Using more different classes could also help, as three were only used for a similar reason as before. This is because each model/class adds time, since the number of models is equal to `(num_classes)(num_models)`, and this also doesn't factor in the time needed to filter and download the dataset. Additionally, instead of fine-tuning, training from scratch (given the time needed to outperform fine-tuning) may strengthen any links in the results.

A third improvement that could be made would be the selection of the classes. These classes were personally chosen based on trying to get three "different-enough" classes, but we could use a objective way to choose the categories by trying to find the most discriminative classes, perhaps using a variation of clustering to see which `num_classes` classes are the furthest from each other. This method will allow for an empirical selection of classes.

---
