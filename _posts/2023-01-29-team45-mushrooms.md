---
layout: post
comments: true
title: Mushroom Classification Project Proposal
author: Drake Cote, Nathan Paredes-Kao
date: 2023-01-29 01:09:00
---

>
<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction - Nathan

Mushrooms are a specific form of fungus that have had their image rise in popular culture as a hip symbol for peace, health, and for their occasional hallucinogenic properties. This has caused a rise in mushroom foraging, a practice of going out into swampy or recently rained on areas to gather mushrooms, as well as commercial mushroom farming where fungal environments are created to grow certain mushrooms for eating. In both of these cases, it is common for multiple types of mushrooms to appear given how easily dispersable fungal spores can be. This can be dangerous as certain mushrooms can appear similar to the untrained eye but can be very poisonous if misclassified. Our goal as mushroom fans ourselves is to develop a model that can help classify mushrooms so that we can continue foraging safely, without having to learn textbooks worth of knowledge to avoid being poisoned. 

### The Dataset - Nathan
Pictures of mushroom

pics of each augmentation would be nice
## Data Engineering - Drake

Mushrooms are generally not found completely isolated on a slate rock, posed for the perfect picture with an even background. They are amidst damp soil, rotting leaves, and all kinds of other foliage. Because of this, we will need to work with our data such that our model can detect where the mushroom is by focusing solely on the shape of the mushroom. Then, we can continue with the mushroom's many various colors and orientations to simply generate more data and not overfit to certain viewpoints. Our biggest challenge is the diversity of ourdataset: we have 89760 images split unevenly amongst 1394 categories. In order to overcome our dataset diversity and small sample size per category, we will attempt to increase our dataset with two augmentation transforms that will aim to make the model focus on shape and simply generate more data through various positional changes. Additionally, because our dataset has so few images for some labels, these augmentations will also help regularize some of the learning to not overfit our limited dataset.

### Aiming for Shape

This transorm contains the grayscale and Random Solarize transformations in an attempt to make the model focus on the shapes of the mushrooms in the image.

#### Grayscale

Grayscaling an image in a basic sense does exactly what it sounds like and makes a colored image have only shades of gray, removing all color. In reality, it collapses the initial three RGB channels into a single channel to remove any indication of color. For the purposes of our dataset we still want a three channel input for the dimensionality of our different models so te image will still hav threee channels but they will be the same where R=G=B. 

#### Random Solarize

On top of grayscaling we will use a tranform called Random Solarization that will invert pixel values above a certain threshold with probability p. The idea behind this transformation is that we don't want the model to learn features solely on edge case pixel values that could highlight bright lights or colors over features of the mushrooms themselves. Therefore, with probability p (set to .9 in our model) we will invert the top most pixel values above the threshold 192 (out of 256 for RGB). We chose the hyperparameters .9 and 192 because we want this transformation to happen often since are concatonating these images with the original data and because 192 is the 75th percentile of pixel values (in general, not calculated over image appearance probability). After these two transformations we will append this grayscaled, solarized dataset onto our original, doubling out dataset size.  

### Increasing Dataset through Position

This transorm contains the random rotation, horizonatal flip, and color jitter transformations in an attempt to augment our data enough to squeeze more information out of our limited dataset.

#### Random Rotation

the Random Rotation augmentation rotates an image randomly between a min and max degree range. We set the range to be 0 to 180. This is an important augmentation particularly because some of the mushroom images in this dataset are always in a certain orientation, i.e. growing straight up vertically versus 45 degrees out from a tree trunk. This is not necessarily because the mushrooms always grow this way, in which case it would be a feature, but are just only photographed from that perepective and therefore we don't want to overfit on just the angle of the stem. Additionally, some of the mushrooms are very flat and turning them a random amount just gives us a new data point from a different orientation the image could've been taken from. 

#### Horizontal Flip

The orizontal Flip augmentation flips the image horizontally with probability p. This augmentation has a very similar purpose to Random Rotation in that it will give us a new perspective and prevent overfitting on certain mushroom orientations common in the dataset.

#### Color Jitter

The Color Jitter augmentation is the last of this series of transforms. Color Jitter randomly changes the brightness, contrast, saturation, and hue of an image. The amount to jitter each factor is chosen uniformly from [max(0,1-factor), 1 + factor]. We chose a brightness factor of .5 because it allowed some of the brighter images to be more similar to other darker images in the dataset and vice versa without making the images too dark or light to see. We set the hue to .3 to jitter the hue similarly in a range that did sometimes drastically change the colors without dramatically warping the image past recognition of shapes from the contrast of shades. We decided not to edit contrast and saturation as in combination with hue and brightness the images were changed too drastically. After these three augmentations we concatenate the transformed data to the previous two datasets, in total tripling our original number of images.

## The Models - Drake
Deep learning has become one of the most popular tools for computer vision and machine learning ever since our computation power increased to the level required to take in the massive amounts of data these models require. Deep Learning models are in a sense exactly how they sound. They are neural networks with many many layers to capture different aspects of data features using backpropogation and series of linear and non-linear transformations to update the learning parameters. We are using several baseline pretrained models with altered output layers for comparison. We extracted the best possible accuracy from Resnet18, Resnet50, VGG16, and ViT with our data. Our goal is to use an ensemble of these different models to try and compensate for our limited dataset, but this goal is gated behind training speed.

### Ensemble - Drake

Individual deep learning networks can be extremely successful at classifying difficult data. How much more so then can a group of these models predict the data together. This is the idea behind an ensemble of models. Each model makes a classification and takes the majority vote betwwen them as the final classification. However, the accuracy of the models we have trained are very different so the regular majority vote did not out achieve our best model by itself. To make up for this imbalance, we can instead do a weighted ensemble where certain models have a stronger vote. We decided to weigh the models by their accuracy with our best model having the highest weight in the vote for the final classification. 

### ViT - Drake

Vision Transformers (ViT) are another model for image recognition that take in an input image as a series of image tokens. They take in each token combined with a positional encoding. This gives the model some initial notion of where the tokens are in relation with one another since they are not just in sequential order like some NLP data that is used in transformers. From here, attention weights are learned between one token to all of the others for each individual token. These attention weights can extract global relationships from the data that can be difficult to get from simple sequential input because they detail the relation between tokens. i.e. how important is this token in the context of another (such as the word he specifically representing to the name Thomas in the sentence "Thomas met a girl named Lucy and he fell in love."). After the multi-head self attention layer residual connections reinput the original token embeddings onto the learned embeddings to complete the pass through the network without passing through any non-linear activations. This process can outperform CNN's significantly in efficiency as pixel arrays and stacked layers of activation functions and convolution are not needed.

add image of self attention
### ResNet - Nathan

### VGG - Nathan

## Results - 

## Conclusion - 

### Code Repositories

[0] [2018 FGVCx Competition Dataset and Repository](https://github.com/visipedia/fgvcx_fungi_comp#data) 

#### Sources

[0] [Mushrooms Detection, Localization and 3D Pose Estimation....](https://arxiv.org/pdf/2201.02837.pdf) Baisa, Nathanael L., and Bashir Al-Diri. Mushrooms Detection, Localization and 3D Pose Estimation Using RGB-D Sensor for Robotic-Picking Applications. Arxiv, 8 Jan. 2022, https://arxiv.org/pdf/2201.02837.pdf. 

[1] Mohanty, Sharada P., et al. “Using Deep Learning for Image-Based Plant Disease Detection.” Frontiers, Frontiers, 6 Sept. 2016, https://www.frontiersin.org/articles/10.3389/fpls.2016.01419/full. 

[2] N, Skanda H. Plant Identification Methodologies Using Machine Learning ... - IJERT. Https://Www.ijert.org/, 3 Mar. 2019, https://www.ijert.org/research/plant-identification-methodologies-using-machine-learning-algorithms-IJERTV8IS030116.pdf. 