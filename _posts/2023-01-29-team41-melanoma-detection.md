---
layout: post
comments: true
title: Skin Lesion Classification
author: Maxwell Dalton (Team 41)
date: 2023-02-26
---


> It is crucial to catch skin cancer at an early stage. This can be done using image classification techniques. Specifically, this post aims to explore using different data augmentation techniques as well as model ensembles to classify skin lesions using the HAM10000 dataset.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Skin cancer can be extremely dangerous, but is usually harmless unless it isn't caught until a later stage, once it has already advanced past the skin alone and into other parts of the body. Because of this, it is crucial to catch it early. With deep learning, it is possible to classify various skin lesions as cancerous or not, which is highly beneficial as this can prevent a doctor's visit (if accurate enough), which would allow for more accessibility. In this project, the goal is to try and build the best model possible using the given resources.

## Dataset
The dataset that I have chosen to work with throughout the course of this project is the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T), which is short for "Human Against Machine with 10000 training images". It is a dataset developed by Harvard Dataverse with 10015 dermascopic training images along with associated metadata. The labels on the data consist of the following skin lesion types:
- Acitinic keratoses and intraepithelial carcinoma
- Basal cell carcinoma
- Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)
- Dermatofibroma
- Melanoma
- Melanocytic nevi
- Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)

### Data Augmentation
Data augmentation is an essential aspect to prevent overfitting to the training data by creating slight variations on the data.

#### Using PyTorch Libraries
Currently, I am using the following data augmentation methods from the PyTorch libraries:
- RandomHorizontalFlip
- RandomVerticalFlip
- RandomRotation
- ColorJitter (slight adjustments to image brightness, contrast, and saturation)
- Normalization (using ImageNet statistics)
Below are some visualizations of before and after the transforms:

![Images Pre-Transform]({{ '/assets/images/team41/pre-transform.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

![Images Post-Transform]({{ '/assets/images/team41/post-transform.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

#### Future: Using GANs
One potential avenue that I wish to explore for the rest of this project is image augmentation using generative adversarial networks (GANs), as described [here](https://arxiv.org/pdf/2004.06824.pdf). 


## The Model

The idea behind the model that I am building is mainly based on the [winning solution to the SIIM-ISIC Melanoma Classification Challenge](https://arxiv.org/pdf/2010.05351v1.pdf) from 2020. Their strategy is that of a model ensemble consisting of a diverse array of pre-trained large models. Furthermore, some of the models utilize the metadata while others ignore it altogether.

### Metadata Models
For the models that take metadata into account, they resemble the following architecture:

![Metadata Model Architecture]({{ '/assets/images/team41/metadata-model.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

My personal model has different metadata associated with it, as it is using a different dataset than the one used in the above architecture, but the main idea is the same: feed the metadata through two linear layers (I use ReLU) before aggregating it with the output from the image that is fed through the CNN, and then finally pass this aggregated tensor into a fully-connected layer to get the class probabilities.

In my experiments thus far, the models that incorporate metadata perform far worse than those that do not. For example, with pre-trained ResNet18, the baseline model can achieve about 65% accuracy on the validation set after two epochs, while pre-trained ResNet18 with metadata shockingly only achieves 22% accuracy on the validation set. Looking into what may be causing this huge discrepancy is one of the main areas that I plan to focus on for the remainder of this project, with one of the ideas I have towards remedying this being to create new metadata features based on the current ones that give more useful information to the model.

### Training Setup
Currently, I am training the models with cross-entropy loss and Adam as an optimizer, with cosine-annealing as the learning-rate scheduler. For the remainder of the project, I wish to look into other areas for these, with a particular focus on trying out [focal loss](https://arxiv.org/pdf/1708.02002.pdf) as opposed to cross-entropy. Focal loss is similar to cross-entropy, but tends to work way better for datasets that have high class imbalance, as it places a higher focus on mis-classified images. 

![Focal Loss]({{ '/assets/images/team41/focal-loss.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}


### Model Ensembling Method
Model ensembling is a great way of avoiding any biases that may be present in a single model. In the ensemble I wish to explore, predictions are first obtained from all models in the ensemble, and then the output probabilities are averaged before making the final class prediction.


---
