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
<div align=center>Fig 2. Mathematical representation of Logistic Regression.</div> 
<br>

The Logistic Regression model is a relatively simple model compared to other complex models like neural networks. There is a single layer of neurons where each neuron computes the weighted sum of the input features and applies the sigmoid function to the result to produce the probability estimate. We apply the sigmoid function to a the simple Linear Regression equation. It is the sigmoid function that maps the weighted sum inputs to the value between 0 and 1 which is the predicted probability that the input belongs in the right class. The common loss function for the Logisitic Regression model is the cross-entropy loss. This loss function measures the difference between the predicted probability distribution and the true probability distribution.  

### Code Implementation
Note: Linear Regression Model
Note: Sigmoid Function + Cross Entropy Loss
Note: Logistic Regression Model
## ResNet18
### Motivation
When it comes to image classification and the Logistic Regression model is a very simple yet quick method when the result is binary. However, this does not mean we cannot use more complex neural network architectures for the task. ResNet18 is a deep neural network architecture that is designed for image classification and is a variant of the ResNet architecture that uses 18 layers including a convolutional layer, four residual blocks, and a fully connected layer. With the introduction to these residual blocks, it removes the vanishing gradient problem because as each layer calculates the gradient layer, it can become exponentially small as our input propagates through each layer. Some reasons why we want to use the ResNet18 model against our Logistic Regression are, 
<ol>
<li>Feature extraction in ResNet18 compared to Logistic Regression's manual feature extraction, can learn hierarchical features from the input image and we want to know how that compares to Logistic Regression.</li>
<li>ResNet18 can also handle more noisy and complex input images compared to Logistic Regression since we can use the multiple layers to extract features.</li>
<li>Performance is the final difference we want to test when compared to Logistic Regression as ResNet18 has previously reached many amazing benchmarks in image classification.</li>
</ol>
### Architecture
![logiregfunc]({{ '/assets/images/team37/resnet18-arch.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
<div align=center>Fig 3. ResNet18 Architecture [2].</div> 
<br>

ResNet18 uses 18 layers residual blocks compared to other neural networks to avoid the vanishing gradient problem. The input changes each layer due the convolutional layers and pooling that occurs during the process. Each convolutional layer is followed by a batch normalization layer and a ReLu activation function. These layers contain four stages which each stage consisting of the residual blocks. Each of these residual blocks contain some convolutional layers with shortcut connections that allow the prevention of the vanishing gradient problem. 
<br> 
<br>
The first convolution layer is the raw input data represented by a 3D vector which is then output with another 3D vector but with a different number of channels. Subsequent layers continue this procedure using the last layer's output as the next layer's input followed by the batch normalization and ReLu activation function. The avgpool layer reduces the height and width of our image classification without changing the number of channels. This helps reduce the spatial dimensions of a feature map while keeping the most important features of our MRI images. The FC layer or fully connected layer, connects every neuron in the previous layer to every neuron in the current layer. For our use case, we use it to map the output of the previous layer to our class labels. 

### Code Implementation
## Result
### Chart Comparison
### Issues
## Demo
Video: [Here](link)

Code Base: [Here](https://github.com/jbaik1/CS-188-CV-Final-Project/blob/main/Brain%20Tumor%20Classifier.ipynb)

## Reference
<ol>
<li>Torres, Renato, et al. ‘A Machine-Learning Approach to Distinguish Passengers and Drivers Reading While Driving’. Sensors, vol. 19, 07 2019, p. 3174, https://doi.org10.3390/s19143174.</li>
    
<li>Ramzan, Farheen, et al. ‘A Deep Learning Approach for Automated Diagnosis and Multi-Class Classification of Alzheimer’s Disease Stages Using Resting-State FMRI and Residual Neural Networks’. Journal of Medical Systems, vol. 44, 12 2019, https://doi.org10.1007/s10916-019-1475-2.</li>
</ol>
[A collection of recent image segmentation methods, categorized by regions of the human body](https://github.com/JunMa11/SOTA-MedSeg)

[A student project from last year in this class. They have specifically studied PDV-Net and ResUNet++.](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2022/01/27/team07-medical-image-segmentation.html)

[A review paper discussing the various methods in deep learning used for medical imaging.](https://link.springer.com/article/10.1007/s12194-017-0406-5#Sec12)

[A medical imaging toolkit for deep learning.](https://github.com/fepegar/torchio/)

[A paper that focuses on data preparation of medical imaging data for use in machine learning.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7104701/)

[Meta analysis of diagnostic accuracy of deep learning methods in medical imaging.](https://www.nature.com/articles/s41746-021-00438-z)

---