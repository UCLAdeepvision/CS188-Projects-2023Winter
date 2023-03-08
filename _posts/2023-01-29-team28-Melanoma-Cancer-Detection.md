---
layout: post
comments: true
title: Melanoma and Skin Cancer Detection 
author: Akhil Vintas and Jeffrey Yang
date: 2023-02-26
---


> One in five Americans develop skin cancer within their lifespans, and roughly 10% of diagnoses is melanoma. Given its degenerative nature, early detection and screening can increase odds of remission. Doctors are still unable to detect if many lesions are cancerous by eye, resorting to invasive biopsies, but with current technologies and data available, neural networks have vasly outperformed doctors in diagnoses. Given how deep learning models have proven to be extremely accurate in the classification of melanoma and other skin cancers, we intend to explore how various models perform and what the strenghts and weaknesses are for each model. Recently, convolutional neural network (CNN) models such as Resnet-50 have achieved over 85% classification accuracy on the ISIC binary melanoma classification datasets. We will explore the relevant high-performing CNN models and their efficacy when utilized for skin cancer classification. We will train these models on varying amounts of data, and determine which models respond most to a change in training data samples. We will also observe how these models respond when they are pre-trained and frozen on data samples. 


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction

Many researchers have already provided strong models to analyze and detect melanoma. Alam Milton analyzed melanoma using an ensemble of neural networks with code provided [here](https://github.com/miltonbd/ISIC_2018_classification). Melanoma detection also has models that can outperform models, such as Pham et al's [here](https://github.com/riseoGit/melanoma-prediction/). Lastly, segmentation is also an area of need to analyze skin images, and Gorriz et. al tackle it [here](https://github.com/imatge-upc/medical-2017-nipsw). We aim to integrate these studies and code repos among others to contribute to a stronger model.

In our project, we set up the data and collab file in this following [link](https://drive.google.com/drive/folders/1e688RzfSggSscRLffN9iBGkpKcn-brcE?usp=sharing).


## Deep Learning CV Models:

### AlexNet

AlexNet is a convolutional neural network with 8 layers. This makes it the least complex model we will utilize, which is noticeable in the relatively low test accuracy. However, this model trains very fast, and is remarkably accurate for its size. AlexNet is the first CNN to win ImageNet. In context, AlexNet can be considered the CNN structure that first popularized CNN’s, which in turn revolutionized the computer vision industry. 

Model Diagram: 

![AlexNet]({{ '/assets/images/team28/AlexNet.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
"*Fig 1. AlexNet: CNN architecture figure* [4]

Model Code:

![AlexNet_Code]({{ '/assets/images/team28/AlexNet_Code.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
"*Fig 1. AlexNet Code: CNN architecture code* [4]

AlexNet consists of 5 convolutional layers followed by 3 linear perceptron layers. The non-linear activation layer used is ReLU, and there are maxpool layers to reduce dimensionality and feature complexity. 

### VGG

VGG is a deep learning CNN model that consists of 16 (VGG-16) or 19 (VGG-19) convolutional layers. VGG-16 achieves over 74% top-1 accuracy on ImageNet, making it a premier classification model, but not the industry standard by any means. It is significantly larger than both AlexNet and ResNet. 

Model Diagram:
![VGG]({{ '/assets/images/team28/VGG.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
"*Fig 1. VGG: CNN architecture figure* [5]

Model Code:
![VGG_code]({{ '/assets/images/team28/VGG_Code.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
"*Fig 1. VGG: CNN architecture code* [7]


### ResNet
ResNet is a convolutional neural network with multiple layers. It is one of the best models in current use, with optimized versions boasting over 84% top-1 accuracy on ImageNet. Previous complicated models such as VGG often ran into what is known as the vanishing gradient problem, which occurs when one variable in a long chain of multiplied chain derivatives is a small value, eventually resulting in a gradient that is near zero. To sidestep this issue, Resnet introduces the idea of residual blocks. 

Block Diagram:
![Resnet]({{ '/assets/images/team28/residual_block.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
"*Fig 1. Resnet: Resnet block architecture figure* [6]

Model Code:
![Resnet_code]({{ '/assets/images/team28/residual_block_code.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
"*Fig 1. Resnet: Resnet code* [6]

The idea behind this residual block is the shortcut, which forwards the output value of a layer x to that of the next layer directly, which allows the gradient to flow through, eliminating the vanishing gradient problem. 

The structure of a ResNet simply consists of numerous convolutional layers, residual blocks, max-pooling, and ReLU. Models range from ResNet-18 to ResNet-152, and they differ in number of layers and other complexity differences. ResNet-50 is the model that we will use in this project. It’s architecture is as follows below:


Model Diagram:
![Resnet]({{ '/assets/images/team28/Resnet.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
"*Fig 1. Resnet: Resnet model architecture* [6]

## Model Comparisons


For our model comparisons, we are using the ISIC 2018 image database. This database consists of 10015 training images, which are all labeled with ground truths in: 

Melanoma (MEL),
Melanocytic nevus (NV),
Basal cell carcinoma (BCC),
Actinic keratosis / Bowen’s disease (intraepithelial carcinoma) (AKIEC),
Benign keratosis (solar lentigo/seborrheic keratosis/lichen planus-like keratosis) (BKL),
Dermatofibroma (DF),
Vascular lesion (VASC)

We will be using the three training models defined above (AlexNet, VGG-16, and Resnet-50) to classify the training images into these 7 categories. We will observe both accuracy and training time on all three of these models. Then, we will experiment on various properties of these three training models. 

### Experiment 1

Vary the amount of training data that is given to each model, and see how each model reacts. We measure the accuracy and training time of each model when given the following number of training samples trained on 20 epochs:

### Experiment 2

Import versions of the three models that have been pre-trained on millions of images from ImageNet. Then, modify these models in one of two schemes: 

Linear: For this scheme, the model will be frozen, then a linear classifier is trained which takes the features before the fully-connected layer. Then, a new fully-connected layer is written, which takes the in-features and outputs scores of size num_classes. 
Finetune: Same as Linear, except that features do not need to be frozen and the model can finetune on the pretrained model.

Performances will be compared between all 3 pretrained models in both Linear and Finetune schemes. All models will be trained on 10015 images, for 20 epochs. Results are as follows: 

### Experiment 3

Experiment 3: So far our experiments have focused largely on measuring the efficacy of the training models, and the emphasis has not been skin-cancer specific. 

<!-- ## Basic Syntax
### Image -->
<!-- Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content. -->
<!-- 
You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
"*Fig 1. YOLO: An object detection method in computer vision* [1]. -->

<!-- 
### Table


### Code Block

### Formula -->



## Reference

[1] Milton, M. "Automated Skin Lesion Classification Using Ensemble of Deep Neural Networks in ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection Challenge." *arXiv*. 2019.

[2] Pham, TC., Luong, CM., Hoang, VD. et al. "AI outperformed every dermatologist in dermoscopic melanoma diagnosis, using an optimized deep-CNN architecture with custom mini-batch logic and loss function." *Sci Rep 11, 17485 (2021)*. https://doi.org/10.1038/s41598-021-96707-8

[3] Gorriz, M., Carlier A., Faure, E., Giro-i-Nieto, X. "Cost-Effective Active Learning for Melanoma Segmentation" *arXiv*. 2017.

[4] https://www.mdpi.com/2072-4292/9/8/848

[5] https://medium.com/mlearning-ai/an-overview-of-vgg16-and-nin-models-96e4bf398484

[6] https://towardsdatascience.com/residual-blocks-buildingda-blocks-of-resnet-fd90ca15d6ec

[7]https://github.com/ashushekar/VGG16


---
