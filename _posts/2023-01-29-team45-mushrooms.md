---
layout: post
comments: true
title: Landslide Detection Project Proposal
author: Drake Cote, Nathan Paredes-Kao
date: 2023-01-29 01:09:00
---

>
<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

### Project Proposal

Mushrooms are a specific form of fungus that have had their image rise in popular culture as a hip symbol for peace, health, and for their occasional hallucinogenic properties. This has caused a rise in mushroom foraging, a practice of going out into swampy or recently rained on areas to gather mushrooms, as well as commercial mushroom farming where fungal environments are created to grow certain mushrooms for eating. In both of these cases, it is common for multiple types of mushrooms to appear given how easily dispersable fungal spores can be. This can be dangerous as certain mushrooms can appear similar to the untrained eye but can be very poisonous if misclassified. Our goal as mushroom fans ourselves is to develop a model that can help classify mushrooms so that we can continue foraging safely, without having to learn textbooks worth of knowledge to avoid being poisoned. 

### Our Focus in Deep Learning
### Data Engineering

Mushrooms are generally not found completely isolated on a slate rock, posed for the perfect picture with an even background. They are amidst damp soil, rotting leaves, and all kinds of other foliage. Because of this we will need to work with our data such that our model can detect where the mushroom is in the image with tools like edge detection and greyscale imaging to focus solely on the shape of the mushroom. Then, we can continue with the mushroom's many various colors. Our biggest challenge is the diversity of ourdataset: we have 89760 images split unevenly amongst 1392 categories. In order to overcome our dataset diversity and small sample size per category, we will be trimming the smallest categories. We will attempt to generate more samples by mirroring and altering the existing images to expand the dataset.

### Deep Learning Models 

Deep learning has become one of the most popular tools for computer vision and machine learning in general since our computation power has increased to the level required to take in the massive amounts of data these models require. Deep Learning models are in a sense just how they sound. They are neural networks with many many layers to capture different aspects of data features using backpropogation and series of linear and non-linear transformations to update the learning parameters. We are using a baseline pretrained Resnet18 model with an altered output linear layer for comparison. This model has a 11.8% validation accuracy after one epoch, but we have found that the training process is extremely slow; one epoch took approximately twenty minutes. We will attempt to increase the speed of training by normalizing the data and using a smaller subset of the data so that we can iterate on our model more quickly. Our goal is to use an ensemble of different neural nets to try and compensate for our limited dataset, but this goal is gated behind training speed. After we have extracted the best possible accuracy from Resnet18 we will try a Vision Transformer as our next model.

### Code Repositories

[0] [2018 FGVCx Competition Dataset and Repository](https://github.com/visipedia/fgvcx_fungi_comp#data) 

#### Proposed Sources

[0] [Mushrooms Detection, Localization and 3D Pose Estimation....](https://arxiv.org/pdf/2201.02837.pdf) Baisa, Nathanael L., and Bashir Al-Diri. Mushrooms Detection, Localization and 3D Pose Estimation Using RGB-D Sensor for Robotic-Picking Applications. Arxiv, 8 Jan. 2022, https://arxiv.org/pdf/2201.02837.pdf. 

[1] Mohanty, Sharada P., et al. “Using Deep Learning for Image-Based Plant Disease Detection.” Frontiers, Frontiers, 6 Sept. 2016, https://www.frontiersin.org/articles/10.3389/fpls.2016.01419/full. 

[2] N, Skanda H. Plant Identification Methodologies Using Machine Learning ... - IJERT. Https://Www.ijert.org/, 3 Mar. 2019, https://www.ijert.org/research/plant-identification-methodologies-using-machine-learning-algorithms-IJERTV8IS030116.pdf. 