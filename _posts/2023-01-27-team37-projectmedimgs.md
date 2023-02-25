---
layout: post
comments: true
title: Medical Imaging
author: Team 37
date: 2022-01-27
---


> Medical Imaging analysis has always been a target for the many methods in deep learning to help with diagnostics, evaluations, and quantifying of medical diseases. In this study, we learn and implement models of Logistic Regression and ResNet18 to use in medical image classification. We use image classification to train our models to find brain tumors in brain MRI images. Furthermore, we will use our own implementation of LogisticRegression and finetune our ResNet18 model to better fit our needs. 

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
We explore different methods of image classification with brain MRI images that may consist of a brain tumor. We train models such as LogisticRegression and ResNet18 and implement them to achieve the best accuracies. The LogisticRegression model was created with our own logic and implementation while the ResNet18 model was finetuned for the task.

## Logistic Regression
### Motivation
Medical imaging is very important when attempting to document the visual representation of a likely disease. Being able to have a larger sample size can help solidify the accuracy the likelihood of diseases. However, it will take a long time and also allow for human error whenever these images are observed by a human. Therefore, it is important for image classification in medical imaging to be as precise and fast as possible. To address this, we use Logistic Regression as a model to accurately and quickly give us the likelihood of a brain tumor. Logistic Regression is quick whenever the response is binary, hence it is a great model to use for our use case. Some challenges that may occur when implementing our design are:
<ol>
<li>High Dimensionality can cause an image to have a large number of pixels depending on our dataset images which can cause overfitting or slow training.</li>
<li>Invariance in our transformations can cause the training model to not take into consideration any images with more variations such as rotated images when image scans are not perfectly straight.</li>
<li>Dataset image dependency is huge here because if there are more images in one image class than the other, there can be an imbalance.</li>
</ol>

### Architecture
![logiregarch]({{ '/assets/images/team37/logireg-arch.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
<div align=center>Fig 1. Simple architecture of the Logistic Regression model [1].</div> <br>

![logiregfunc]({{ '/assets/images/team37/logireg-func.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
<div align=center>Fig 2. Mathematical representation of Logistic Regression.</div> <br>

The Logistic Regression model is a relatively simple model compared to other complex models like neural networks. There is a single layer of neurons where each neuron computes the weighted sum of the input features and applies the sigmoid function to the result to produce the probability estimate. It is the sigmoid function that maps the weighted sum inputs to the value between 0 and 1 which is the predicted probability that the input belongs in the right class. 

### Code Implementation
## ResNet18
### Motivation
### Architecture
### Code Implementation
## Result
### Chart Comparison
### Issues
## Demo
Video: [Here](link)

Code Base: [Here](link)

## Reference
<ol>
<li>Torres, Renato, et al. ‘A Machine-Learning Approach to Distinguish Passengers and Drivers Reading While Driving’. Sensors, vol. 19, 07 2019, p. 3174, https://doi.org10.3390/s19143174.</li>
</ol>
[A collection of recent image segmentation methods, categorized by regions of the human body](https://github.com/JunMa11/SOTA-MedSeg)

[A student project from last year in this class. They have specifically studied PDV-Net and ResUNet++.](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2022/01/27/team07-medical-image-segmentation.html)

[A review paper discussing the various methods in deep learning used for medical imaging.](https://link.springer.com/article/10.1007/s12194-017-0406-5#Sec12)

[A medical imaging toolkit for deep learning.](https://github.com/fepegar/torchio/)

[A paper that focuses on data preparation of medical imaging data for use in machine learning.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7104701/)

[Meta analysis of diagnostic accuracy of deep learning methods in medical imaging.](https://www.nature.com/articles/s41746-021-00438-z)

---