---
layout: post
comments: true
title: Object Detection and Classification of Cars by Make using MMDetection
author: Ben Klingher, Erik Ren
date: 2023-01-29
---

<!--more-->

# Object Detection and Classification of Cars by Make using MMDetection

## Abstract

The detection and classification of vehicle make and model in images is a challenging task with various practical applications, including parking administration, dealership stock management, surveillance systems, and civil security. In this project, we used the MMDetection library to build models for detecting and classifying the make of cars trained on annotated images from the Stanford Cars dataset. Because this dataset dates from 2013, it displayed issues generalizing to more modern cars. Other challenges arose in the approach to finetuning the different detector models and with the large number of classes we were attempting to classify. To address these challenges, we explored various methods, including limiting the number of classes to just the make of a car, using pre-trained (COCO dataset) object detectors and finetuning them to detect car makes, and, finally, creating a custom Resnet model (pretrained on Imagenet) to apply on top of an existing pretrained COCO detector. Our experiments revealed that finetuning pretrained detector models alone did not deliver satisfactory results, and we achieved better performance by limiting the number of classes and adding a secondary classification model, after detecting the bounding box for the cars with a model better trained to that application.

## Introduction

We leveraged the MMDetection library to train a model to detect and classify the make of cars using the Stanford Cars dataset with the goal of applying it to real-life photos from the UCLA neighborhood. Although we were able to develop a successful model to detect and classify the makes of cars, this task presented significant challenges including the relatively large number of classes (combinations of make model and year) and the imbalanced class distribution, as well as the deep subtlety needed to identify specific car models.

The Stanford Cars dataset consists of 16,185 images each annotated with bounding boxes labeled into one of 196 classes, which indicate the make, model, and year for the car. While the dataset is roughly balanced in the specific classes of make, model and year, it is particularly unbalanced with respect to just the make of the vehicle, with some makes having a substantially smaller number of samples than others. It is also very spotty on the coverage of vehicles encountered in public. For instance, it does not have any Subaru cars in the dataset. Another significant challenge was the outdated nature of the Stanford Cars dataset. The dataset was last updated in 2013, making it difficult to accurately detect the make of newer cars in current day photos/videos. Moreover, the dataset's images were taken in a controlled environment, often in dealerships, making it challenging to generalize the trained model to real-world scenarios.

Many of the models we employed in this project were pretrained, so it is useful to discuss the contents of those datasets. The COCO dataset is an object detection dataset which includes annotations for a wide variety of recognition tasks, including object detection and classification. It categorizes objects into 80 different categories that range from a wide variety of different objects, from vehicles to foods. The Resnet model we ultimately trained for classifying the make of cars is pretrained on Imagenet, which is a familiar dataset.

Early on in this project, we discovered that classifying make/model/year of cars is a difficult task. Even many humans struggle with this. We encountered a new issue when we realized the very different types of classification that the pretrained models performed versus the classification necessary to differentiate specific vehicle models. The object detection models that we employed in the MMDetection library were trained primarily on the COCO dataset. In that dataset, object classes vary from “cars” to things like “cows”, “spoon”, “banana”, etc. The types of features that a convolutional neural network would extract to perform that kind of classification are very different from those necessary to identify even the make, let alone the model and year, of a car. In the latter case, much finer details of the car itself are necessary to make those kinds of distinctions.

First we attempted to use the pre-trained detection models and fine-tuning techniques, but found that pre-trained models alone did not deliver satisfactory results due both to the dataset's large class distribution and the nature of the features found in the pretrained model. Consequently, we explored several methods to handle these challenges including limiting the total number of classes to just the make of a car and creating a custom Resnet model to further improve overall performance. We also explored several other car datasets like the Car Connection dataset in order to include more recent car models that were not present in the Stanford Cars dataset.

To address these challenges, we iterated on different approaches to the problem, from fine tuning existing object detection methods, to training an independent classification model. We provide a detailed account of our approach to addressing the challenges of car make classification and describe our model selection and training process. We also discuss our efforts to handle the issues with the low performing model and how we re-organized the data and our model architecture to greatly improve our results. Furthermore, we present the results of our experiments, including an evaluation of the effectiveness of our approach in accurately detecting the make of cars in real-life photos from the UCLA neighborhood.

## Method

Our initial approach was to use a pre-trained object detector in MMDetection that had been trained on the COCO dataset and fine-tune it to the Stanford Cars dataset. In addition to the task of accurately generating the bounding boxes for the cars, we aimed to identify the make, model, and year of the cars depicted in each image. 

The Stanford Cars dataset includes bbox annotations, so we ran a long series of training cycles using its annotations to see if we could correctly detect and label the make/model/year of these images. We tried several different models pretrained on the COCO dataset. Our initial experiment was with one version of the Faster R-CNN model. We also tried using a version of Mask R-CNN with the mask head turned off. Ultimately, we had the best results using the small version of YOLOX provided by MMDetection.

Training was complicated by the size of the dataset and the limited GPU memory. After an initial period of testing using a mini dataset, we then ran eight hour periods of training for each of these models. In all of our initial trials, these resulted in extremely poor results. All the mAP values for validation were near zero and running inference on test, or even training, images resulted in no bounding box found at all, or only boxes with very low confidence scores.

Eventually we decided that the 196 classes of make/model/year were too many categories to accurately train a fine tuned model. So, we decided to move to training only to classify by the make of the vehicle. This approach reduced the number of classes to 49, thereby improving the model's performance slightly. Using YOLOX as our initial pretrained model, we trained for over 20 hours on the Stanford dataset, and did begin to see some reasonable results. However, the model still struggled to return results for many test images. A particular problem was the bounding boxes output by the model. Even if the classification of the make was correct the bounding box often failed to correctly identify the care in the image.

This led us to a new consideration. These models pretrained on the COCO dataset are already particularly good at drawing bounding boxes around cars in a wide variety of images. Therefore, the logical approach would be to use the pretrained detection model to generate the bounding boxes which we would classify with a secondary classifier, trained independently.

So, we decided to train a custom Resnet model to classify the make of the cars. We tried various approaches to this classification model. We began by training a Resnet50 on the classification task. We tried both Resnets pretrained on Imagenet and fresh Resnet models. We found that those pretrained on ImageNet reach satisfactory performance far more quickly than those without the pretraining. The Resnet50 was able to achieve a test accuracy of about 60% in classifying the make of vehicles, which was too low for our goals. We then trained a Resnet10 pretrained on ImageNet and we were able to reach a test accuracy of 75% for classifying the make of the vehicles.

Despite the promising results of our custom Resnet model, its performance on newer cars in real-life photos was not consistent. This issue arose due to the limitations of the Stanford Cars dataset, which contained only images up until the year 2013, leaving out a substantial portion of cars that are commonly found on roads today. To overcome this challenge, we attempted to expand our dataset size by incorporating additional data from the Car Connect dataset, which includes more recent images of cars. One issue with this dataset was the type of car images included; many of the images were of the interior of the car which is not useful for our task. This required us to filter out the images in the dataset by using YOLOv3 to filter out the interiors of cars. By doing so, we were able to collect a few thousand more images of newer cars, but it was not significant enough to improve our performance. Due to the issues of merging these two datasets and the failure to deliver increased performance, our final Resnet model was trained only on the Stanford Cars dataset.

We also experimented with different thresholds for rejecting the labels. If the YOLOX model did not have at least 70% confidence that the object was a car then it was not labeled. Because of the number of classes in our model, we often had lower confidence levels in the classification stage, so we accepted labels with a confidence of at least 30% in our examples shown in this report. Even this low confidence limit has been shown to be useful, as will be demonstrated below.

In the end, our best performing architecture used a YOLOX model pretrained on the COCO dataset to draw the bounding box around the vehicles, which is then passed to the Resnet classifier to identify the make of the vehicle in the bounding box. 

## Results

ADD MORE RESULTS BELOW FROM OTHER TRIALS:
Our top performing architecture yielded promising results with a validation score of around 75% for our custom Resnet model on the Stanford Cars dataset. The average precision at IoU=0.5 for the final model was around 0.5 based on annotations in the validation set. The fairly low average precision was mainly due to the large number of incorrect labels. Because our classification based architecture returns a label for each bounding box as long as it achieves a minimum of 30%, we are skewed to produce more false negative labeling. 

Here we will compare several results from the fine tuned YOLOX architecture, to the stacked classifier approach.

![Audi]({{'/assets/images/team49/audi_resnet.png'|relative_url}})

Though performing fairly well on the test data, we experienced challenges with the accuracy of our model on real-life photos, particularly with newer cars. This can be attributed to the outdated nature of the Stanford Cars dataset, which only included cars up until the year 2013, leaving out a significant number of cars commonly seen on roads today.
Bad on video, bad on modern cars


## Discussion

This should be the most important part of this report. You can discuss insights from your project, e.g., what resulted in a good performance, and verified/unverified hypotheses for why this might be the case. Explain the ablation studies you run and the observations. Some visualization could help!

Advantages of fine tuned: higher threshold to offer a prediction, so less often wrong

Harder than expected to classify cars (compare to the way humans do it), fine tuning detector was not the best approach (but better sometimes at labeling), YOLOX on COCO good for boxes, fine tuning with imagenet pretrained Resnet was much better than starting from scratch, limitations of the dataset, makes sense to use model already trained to identify cars and then stack a classifier than to train a detector/classifier together, COCO already good at identifying cars

COCO fintuned, often good at classification, but bad at boxes, always near zero mAP score

COCO + Resnet better at boxes: higher map of ~.5

Future results: real time, better/more modern car dataset, longer training on finetuning, training on more wild data (instead of mostly dealership or car lots)


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

Deep Learning Based Vehicle Make-Model Classification
https://arxiv.org/abs/1809.00953

Unsupervised Feature Learning Toward a Real-time Vehicle Make and Model Recognition
https://arxiv.org/abs/1806.03028

