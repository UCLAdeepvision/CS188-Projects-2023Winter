---
layout: post
comments: true
title: Object Detection Algorithms
author: Anubha Kale, Ellen Wei
date: 2023-02-26
---

> Object detection is an invaluable computer vision task that requires locating objects in an image and classifying or identifying them. This project explains some deep learning algorithms used for object detection and explores transfer learning. Using OpenMMLab's MMDetection toolbox, we finetune various pre-trained models on custom datasets and compare the results. 

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---
## Introduction
### What is object detection?
Object detection is a computer vision task that requires locating objects in a digital photograph and classifying or identifying them. Given an image as input, the object detection model should output a set of annotations. Each annotation contains the bounding box, represented by a point, width, and height, the class label of the object in this bounding box, and a confidence score indicating the certainty with which the model has classified this object [citation](https://www.image-net.org/challenges/LSVRC/2017/index.php). *Figure 1* below shows a sample output of an object detection model. Before the task of object detection, computer vision researchers focused on image classification, that is, classifying an image in one or more categories. Object detection can be considered a more advanced extension of image classification, in which we label specific portions of the image, instead of the entire image. This project explains some major deep learning algorithms used for object detection and reproduces their results using OpenMMLab's MMDetection toolbox.

![AltText]({{ '/assets/images/team20/Detected-with-YOLO.jpg' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Figure 1. An example output of an object detection algorithm called YOLO* [1].

### Applications of Object Detection
Object detection has many applications. Self-driving cars use object detection to detect locations of other cars, pedestrians, traffic signals, and other critical objects on the road. More generally, robots use object detection to view and interact with the world around them. Another use of object detection is to caption images and videos based on the objects in them. This is needed for increasing web accessibility. Facial detection is also a subset of object detection focused on detecting where faces are located in an image. Facial detection is used in cameras, photography, and for marketing. Video surveillance makes use of object detection for security purposes. These are just some of the numerous applications of object detection.

### Major Milestones in History of Object Detection
Over the past 25 years, object detection has grown greatly as a field of computer vision research. The figure below shows the number of publications in object detection over time [citation: Object Detection in 20 Years: A Survey](https://arxiv.org/pdf/1905.05055.pdf). 

![AltText]({{ '/assets/images/team20/num_papers_over_last_20_years.jpg' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Figure 2. Increase in Number of Object Detection Publications From 1998 to 2001* [1].

The history of object detection can be divided into two eras: pre and post deep learning. Before 2014, object detection was done using traditional methods, including handcrafted, low-level features, and shallow machine learning architectures. After 2014, deep learning became the focus of object detection algorithms research. Here is a list of important object detection milestones, describing how approaches changed over time:

* 2001: Viola Jones Detectors detect human faces in an image by sliding windows of all possible sizes over all locations of the image, and checking whether these windows contained a face, using a machine learning classifier. The researchers incorporated new techniques to speed up performance. Viola and Jones used Haar-like features, commonly used for object detection at the time. Haar-like features are adjacent rectangles that are white and black, or of drastically different intensities, that are used to identify common patterns in images. For example, they are used to detect edges and certain regions of the face like the eyes and nose, that have contrasting dark and light areas. [citation: Object Detection in 20 Years: A Survey](https://arxiv.org/pdf/1905.05055.pdf) [citation: Rapid Object Detection using a Boosted Cascade of Simple Features](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf). 

* 2004: Scale Invariant Feature Transform SIFT is a method to extract features from an image that are invariant to scale and rotation of the image. Using SIFT features, variations in illumination, perspective, and noise does not greatly affect the accuracy of image matching. [citation: paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf).

* 2005: Histogram of Oriented Gradients (HOG) feature extraction method is proposed, which outperforms other methods including SIFT. [citation: HOG paper](https://hal.inria.fr/file/index/docid/548512/filename/hog_cvpr2005.pdf). 

* 2008: Deformable Part-based Model (DPM) was the quintessential traditional object detection model. DPM used a divide-and-conquer approach, called the star-model. It would detect different parts of the object in order to detect the object as a whole. [citation: Object Detection in 20 Years: A Survey]

* 2012: Imagenet classification with deep convolutional neural networks. A deep convolutional neural network is able to learn high-level features automatically. 

* 2014: Regions with CNN Features (R-CNN). 2014 saw the start of the Deep Learning Era of Object Detection. Inspired by the use of deep convolutional neural nets for image classification, researches tried using CNNs for object detection. R-CNN proposed a succesful architecture that saw many future improved versions over the next several years. R-CNN is discussed in detail later in this article. 

* 2015: Fast R-CNN
* 2015: Faster R-CNN
* 2015: You Only Look Once (YOLO)
* 2015: Single Shot MultiBox Detector
* 2017: Feature Pyramid Networks (FPN)
* 2020: Object Detection with Transformers

## Basic Definitions
### Convolutional Neural Networks

Before exploring models in depth, we must provide a background to the building blocks and terminology. Convolutional neural networks (CNN) are a class of artificial neural networks that are most commonly applied to analyze visual image. They are specifically designed to process pixel data and the connectivity pattern between neurons is loosely inspired by biological processes resembling the organization of the animal visual cortex. They are based on convolution filters, also known as kernels, to produce feature maps. A mathematical operation called a convolution is used in place of general matrix multiplication to generate the layers. The network learns to optimize these filters through automated learning.  

A CNN consists of multiple layer elements: an input layer corresponding to the image dataset, convolutional layers where the image becomes abstracted to a feature map, pooling layers to reduce dimensions by combining small clusters, and fully connected layers to connect every neuron in one layer to every neuron in another layer. Each neuron inputs from only a restricted area of the previous layer called the receptive field. Weights are the vector of weights and biases are called filters and represent particular features of the input.

### Performance Metrics
Before discussing different object detection algorithms, it is important to discuss the metrics used for evaluating their performance. The standard metric used in object detection research is mean Average Precision (mAP). The terms below help explain how we determine whether a prediction is correct or incorrect, and the mAP metric for model performance.

#### Precision
Precision is a value representing the percentage of correctly identified objects out of all objects the model identifies. A model with high precision is desirable.

$$Precision = True Pos/(True Pos + False Pos)$$

#### Recall
Recall measures the ability of the model to detect all ground truths, in other words, all objects in the images. A model with high recall is desirable.

$$Recall = True Pos/(True Pos+False Neg)$$

#### Intersection over Union (IoU)
IoU evaluates the degree of overlap between the ground truth and prediction and ranges from 0 to 1, with 0 meaning no overlap and 1 indicating perfect overlap. The IoU of the prediction bounding box, P, and the ground truth bounding box, G, is defined as the area of intersection of P and G areas, divided by the area of union of the P and G areas. The image below shows a visualization of computing IoU. 

![AltText]({{ '/assets/images/team20/map-iou-figure.jpg' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Figure 3. Visualization of IoU Computation* [1].

For a given class label, say **bird**, if the prediction bounding box and the ground truth bounding have an IoU of greater than 0.5, the prediction is considered correct. Different IoU thresholds (denoted by $$\alpha$$), like 0.75 can be chosen if we want to increase the requirement of overlap for a prediction to be deemed correct. If multiple predictions are made for the same object, the first one is counted as a correct prediction, while the rest are considered incorrect, because it is the job of the algorithm to filter out duplicate predictions of the same object. If the model labels the same bird using 10 different boxes, varying only in location and size, only 1 prediction will be counted as a correct prediction, while the other 9 will be counted as incorrect predictions.  

#### Precision Recall Curve
Precision recall curve is a plot of precision and recall at varying confidence thresholds. Each confidence threshold (not to be confused with IoU threshold $$\alpha$$) is used to determine whether a prediction should be trusted or not. The model outputs a confidence score with each prediction, indicating how confident the model is about this prediction. For a given confidence threshold, we can calculate the precision and recall scores, with only predictions above the threshold considerd as positive predictions. If the threshold were 0.70, then a prediction with 0.60 confidence would not be considered as a positive prediction. Each threshold creates a point on the precision recall plot, with many of these points forming the curve. 

#### ROC Curve
A Receiver Operating Characteristic (ROC) curve is a graph showing the performance of a model at a all classification thresholds. It plots the two parameters: true positive rate (equivalent to Recall) and false positive rate.  

$$TPR = True Pos/(True Pos+False Neg)$$  

$$FPR = False Pos/(False Pos+True Neg)$$

#### Average Precision (AP)
Average Precision is the area under the precision recall curve. We calculate AP for each class in the dataset. $$AP@\alpha$$ denotes an AP that is calculated using an IoU threshold of $$\alpha$$.

#### Mean Average Precision (mAP)
The mean Average Precision is the AP averaged over all classes in the dataset (e.g. classes = [bird, cat, dog, person, car, ...]).  

To calculate the average precision for the class, **bird**, in our dataset, we first sort the predictions by their confidence levels. For each confidence level, we calculate the precision and recall scores of all predictions above that confidence level. We plot (recall, precision) points to form a precision-recall curve. The area under the curve gives the Average Precision for the **bird** class. Then, we find this precision for each class in the dataset, and the mean of all the AP values gives the mean Average Precision, or mAP, of predictions. 

## R-CNN: Regions with Convolutional Neural Networks
### Motivation
From 2010 to 2012, object detection performance was generally stagnant, with little improvements made by ensemble models [Citation: Rich feature hierarchies for accurate object detection and semantic segmentation]. The best models up to this point were made using SIFT and HOG, low-level, engineered features. The R-CNN object detection algorithm resulted in a mAP of 53.3% on the PASCAL VOC 2012 dataset, which was a 30% improvement compared to the previous best performance [Citation: Rich feature hierarchies for accurate object detection and semantic segmentation]. In 2012, [A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet classification with deep convolutional neural networks] showed that using CNNs for image classification resulted in a significant performance improvement. Therefore, it made sense to try using CNNs for the task of object detection. 

### Main Ideas
One main problem to solve in object detection is localization, the task of locating objects in the image. This can be done by treating localization as a regression problem or using a sliding-window approach, both of which have certain issues with performance for deep neural networks. R-CNN uses region proposals, which are bounding boxes of areas of interest in the image, that are suspected to contain an object to be labeled. Given an input image, it generates about 2000 category-independent region proposals, of various sizes, and uses a CNN to create a fixed-length feature vector for each region proposal. Then, each region proposal is classified using class-specific linear SVMs. The combination of CNNs and region proposals explains the model's name, *R-CNN*. The figure below illustrates this process at a high level. 

![AltText]({{ '/assets/images/team20/r-cnn-figure1.jpg' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Figure 4. R-CNN steps at a high level* [1].

Another major problem to solve is the insufficient amount of labeled data for training a large CNN. One solution to this problem is to do unsupervised pre-training, and then supervised fine-tuning. The R-CNN paper proposes using supervised pre-training on a large auxiliary dataset, and then domain-specific fine-tuning on a small dataset, as a successful method of training deep CNNs when lacking adequate training data.

R-CNN consists of 3 parts.
1. Generate Category-Independent Region Proposals
R-CNN is compatible with many different region proposal methods, but uses selective search for this paper. Selective search combines aspects of exhaustive search and segmentation to create a stronger region proposal algorithm. It uses a bottom-up approach, finding many small, separate areas of the image, then combining them in a hierarchical manner to find the final region proposals. While this is not a deep learning algorithm, it is still important for region proposals in object detection. 

2. Use CNN to Extract a Fixed-Length Feature Vector From Each Region Proposal
A CNN is used to create a 4096-dimensional feature vector from each region proposal. A mean-subtracted 227x227 RGB image is forward propagated through 5 convolutional layers and 2 fully connected layers. To convert the region proposal, an arbitrary sized image, to a 227x227 image, warping is used to resize the image. To do testing, the test image goes through selective search to produce about 2000 region proposals, which are then fed through the CNN to produce feature vectors. 

3. Class-Specific Linear SVMs Used for Classification of Region Proposals
Finally, each feature vector is scored by each class-specific SVM, indicating the likelihood of the region proposal being in the SVM's class. For each class, duplicates are removed by rejecting boxes that overlap largely with a higher scored box.

### Efficiency of R-CNN 
Because the only class-specific step is step 3, the method outlined by R-CNN is very efficient. The CNN model parameters are shared across classes, and the CNN creates a relatively small feature vector (4K dim). The computation of region proposals and feature vectors is only done once, shared by all classes. The feature matrix is 2000x4096, with 2000 region proposals each encoded as a 4096 feature vector. The SVM weight matrix is 4096xN, since there is a separate SVM for each of N classes. 

### Training the R-CNN using Innovative Approach
Pre-training of the CNN was done using the auxiliary dataset, containing only image-level labels, not bounding box information. Then, the CNN model parameters are used as initializations for the domain-specific CNN training. To fit the CNN to our object detection task, as opposed to image classification, the CNN's classification layer is modified to be for N+1 classes, where N is the number of object classes and +1 is for the background class, instead of the unrelated class size of the image classification dataset. We use the warped region proposals to complete the training of the CNN. For the Stochastic Gradient Descent, each mini-batch during training contains 32 positive warped region proposals and 96 background ones, in order to favor the positive windows, which are the minority group. This supervised pre-training on a large auxiliary dataset and then domain-specific fine-tuning on a small dataset was an innovative approach that tackled the issue of a lack of training data.

### Summary
Overall, R-CNNs were a major breakthrough in improving object detection model performance by introducing a method of applying CNNs to region proposals, and a method of training with limited object detection data by using auxiliary image classification data.

## Fast R-CNN
Fast R-CNN trains and predicts faster and with higher accuracy than R-CNN. In R-CNN, the generated region proposals are too general, therefore, after the SVMs do classification, a bounding-box regression stage is used to make the bounding boxes more precise. Training the SVMs and bounding-box regressor is expensive in both compute time and space in memory because of the large feature vectors representing the region proposals.  

Fast R-CNN aims to simultaneously learn to classify region proposals and refine the object localizations in its fully connected layers. Fast R-CNN training can be done in a single-stage and updates all the network layers, uses a multi-task loss, and requires no disk storage for feature caching.  

![AltText]({{ '/assets/images/team20/fast-r-cnn-architecture.jpg' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Figure 5. Fast R-CNN Architecture* [1].

The main difference in Fast R-CNN is that the entire image is passed through the CNN to get a feature map representing the entire image. This is very different from R-CNN, where each region proposal is passed through the CNN to get features for the region proposals. This difference means that Fast R-CNN requires less passes through the CNN, feeding-forward once per image, instead of 2000 times per image, assuming there are 2000 region proposals for an image. For a given image, for each object proposal, a fixed-length feature vector is extracted from the feature map of the image. This is done by a Region of Interest (RoI) pooling layer.  

These feature vectors are then passed through some fully connected layers, and split to result in two outputs. One split goes through softmax to give the vector probabilities of the region proposal being in each of the classes and the additional *background* class. The other outputs the bounding boxes for each class, using per-category bounding-box regressors. Figure below shows the Fast R-CNN architecture. citation: R-CNN paper.  

## Faster R-CNN
Faster R-CNN further improves the train and prediction time of the object detection model by introducing Region Proposal Network (RPN) instead of pre existing region proposal algorithms such as Selective Search and EdgeBoxes. These pre existing methods were found to be bottlenecks, accounting for at least half of the inference time in state-of-the-art object detection models. Figure 6 shows the architecture of Faster R-CNN.

![AltText]({{ '/assets/images/team20/faster-r-cnn-architecture.jpg' | relative_url }})
{: style="width: 300px; max-width: 100%;"}
*Figure 6. Faster R-CNN Architecture* [1].

Two main contributions of Faster R-CNN are:
1. Region Proposal Network (RPN)  
 The RPN is a deep convolutional neural network to compute the region proposals for a given image. It shares its layers with the CNN used in R-CNN and Fast R-CNN to compute the features. This sharing means that the RPN does not add significant time to the total inference time, because it uses the computation that already needed to be done to get the convolutional feature map. The RPN takes an image as input, and outputs a set of region proposals, each with an objectness score representing how likely the region is to contain any object, as opposed to it being in the *background* class. 

2. Anchor Boxes  
 Faster R-CNN introduces anchor boxes to generate rectangular region proposals of different shapes and sizes. Using a fixed-size sliding window over the convolutional feature map, each window is passed through classifier and bounding-box regressor. At each sliding window location, multiple region proposals are predicted, with various scales and aspect ratios. We call these anchor boxes, because they are boxes centered at the sliding window's center, called the anchor. The figure below shows the RPN and visualizes the anchor boxes given the current sliding window position.

![AltText]({{ '/assets/images/team20/faster-r-cnn-rpn.jpg' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Figure 7. Faster R-CNN's Region Proposal Network with Anchor Boxes* [1].


## YOLO

The YOLO (You Only Look Once) algorithm is a real-time object detection framework first introduced in 2015. It identifies spatially separating labeled bounding boxes which correspond to rectangular shapes around the objects. It has gone through multiple improvements through the years with YOLOv7 (released in 2022) making significant improvements in the field. The latest version, YOLOv8, was released in January 2023.  

YOLO is popular due to its speed, detection accuracy, good generalization abilities, and open-source nature. The YOLOv1 architecture consisted of 24 convolutional layers, 4 max-pooling layers, and 2 fully-connected layers. It works based on the following 4 approaches: residual blocks, bounding box regression, IoU, and non-maximum-suppression.  

YOLOv2 released in 2016 made use of Darknet-19 as a new architecture and implemented convolution layers using anchor boxes. This simplified by replacing the fully connected layers with anchor boxes instead of predicting the exact coordinates of bounding boxes. YOLOv3, released in 2018, included Darknet-53 (106 neural network with upsampling networks and residual blocks) as a new architecture. This performed 3 predictions at different scales for each location, used a logistic regression model to predict the 'objectness' score for each bounding box, and had more accurate class predictions. YOLOv4 used a backbone with CSPDarknet-54 (29 convolution layers with 3x3 filters, ~27.6 million parameters) and instead of a Feature Pyramid Network, it uses PANet for parameter aggregation from different detection levels. YOLOv5 used CSPDarnet53 as the backbone and reduced the number of layers and parameters, increasing both forward and backward speed. YOLOv7 made a major change in the architecture and used the trainable 'bag-of-freebies' model, allowing the model to improve the accuracy without increasing thee training cost. It uses E-ELAN as a backbone. In January 2023, YOLOv8 was released.

## Transfer Learning
(to be continued)  

We aim to analyze how well different pre-trained object detection models perform when finetuned to perform object detection on new datasets. 

### Datasets
One dataset we have chosen is the [Aerial Maritime dataset](https://public.roboflow.com/object-detection/aerial-maritime).   

### Code
[Link to Colab Notebook](https://drive.google.com/file/d/19ZgCCU6J6CSmkXC5asnsnd2DQrTkA1uI/view?usp=sharing).  

### Results

1. 
* Model: Faster R-CNN 
* Pretrained on: COCO 
* Finetune epochs: 2
* mAP@50: 0.519
* mAP@75: 0.259 

![AltText]({{ '/assets/images/team20/faster-r-cnn-2-epochs-acc-loss-plots.jpg' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Figure X. Train accuracy and loss curves over time during 2 epochs of finetuning the Faster R-CNN originally trained on COCO.* [1].

## Relevant Papers

1. You Only Look Once: Unified, Real-Time Object Detection (YOLO)
   - [Paper](https://arxiv.org/abs/1506.02640)
   - [Code](https://pjreddie.com/darknet/yolo/)
2. Rich feature hierarchies for accurate object detection and semantic segmentation (R-CNN)
   - [Paper](https://arxiv.org/abs/1311.2524)
   - [Code](https://github.com/rbgirshick/py-faster-rcnn) (for Faster R-CNN)
3. MMDetection: Open MMLab Detection Toolbox and Benchmark
   - [Paper](https://arxiv.org/abs/1906.07155)
   - [Code](https://github.com/open-mmlab/mmdetection)
4. Object Detection Overview Paper
   - [Object Detection with Deep Learning: A Review](https://arxiv.org/abs/1807.05511)

## Reference

\[1] J. Redmon, S. K. Divvala, R. B. Girshick, and A. Farhadi, ‘You Only Look Once: Unified, Real-Time Object Detection’, _CoRR_, vol. abs/1506.02640, 2015.

\[2] R. B. Girshick, J. Donahue, T. Darrell, and J. Malik, ‘Rich feature hierarchies for accurate object detection and semantic segmentation’, _CoRR_, vol. abs/1311.2524, 2013.

\[3] K. Chen et al., ‘MMDetection: Open MMLab Detection Toolbox and Benchmark’, _CoRR_, vol. abs/1906.07155, 2019.

\[4] Z.-Q. Zhao, P. Zheng, S.-T. Xu, and X. Wu, ‘Object Detection with Deep Learning: A Review’, _CoRR_, vol. abs/1807.05511, 2018.

---
