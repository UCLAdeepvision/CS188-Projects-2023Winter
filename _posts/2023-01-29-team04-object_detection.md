---
layout: post
comments: true
title: Object Detection Algorithms
author: Karl Goeltner, Rudy Orre (Team 04)
date: 2023-01-29
---


> Topic: Object Detection Algorithms


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Object Detection Algorithms
For our project, we have chosen to focus on object detection algorithms. We are interested in detecting everyday objects in a variety of settings. We want to compare the different objection detection algorithms and determine their weakpoints. To innovate, we are interested in creating our own dataset focused on these weakpoints or perhaps adding our own model training methods or adjusting hyperparameters to adjust the algorithm to function in different scenarios.

### MMDetection Faster R-CNN Model
MMDetection is a popular object detection toolbox which is a part of the OpenMMLab project. We focus on Faster R-CNN, which is a state-of-the-art object detection model that combines the Faster R-CNN framework with a Region Proposal Network (RPN) for improved accuracy and speed. The model is trained on large datasets such as COCO and VOC to detect objects in images with high precision and recall. The Faster R-CNN model consists of two modules:
<ol>
  <li>Region Proposal Network (RPN) - which generates a set of candidate regions of interest in the image</li>
  <li>Classifier Network - labels each region as belonging to a particular object class or as background</li>
</ol> 
The model achieves its fast detection speed by using a shared feature map for both the RPN and classifier networks, allowing it to perform object detection in real-time applications. Overall, MMDetection Faster R-CNN is a powerful object detection model that has demonstrated excellent performance on various benchmark datasets.

### YOLOv3 Model
YOLOv3 (You Only Look Once version 3) is a popular object detection algorithm developed by Joseph Redmon and Ali Farhadi in 2018. It is a deep neural network-based approach that can detect objects in images and videos with high accuracy and real-time speed. YOLOv3 is an improvement over the earlier versions of YOLO, which had lower accuracy and slower processing speeds. For this project, we are implementing it in Pytorch such that we can apply different transfer learning techniques, using pre-trained weights on the COCO dataset. This implementation can be found at https://github.com/rudyorre/yolo-v3.

### COCO Dataset
The COCO (Common Objects in Context) dataset is a large-scale object detection, segmentation, and captioning dataset that was introduced in 2014. It contains over 330,000 images, each of which has been labeled with information about the objects contained within it. The dataset includes 80 different object categories, such as people, animals, vehicles, and household objects. Since the YOLO model is pre-trained on the COCO datset, it will be one of the main datasets used for experimenting with.

### Metrics
To evaluate our models and to compare their respective performances, our main metric will be the Mean Average Precision (mAP). It is a common evaluation metric used in object detection and recognition tasks. It is used to measure the accuracy of an object detection algorithm by comparing the predicted bounding boxes for objects in an image or a set of images with the ground truth bounding boxes. In general, higher mAP values indicate better object detection performance, so we'll be using it to see good (or bad) our models perform, especially after transfer learning on a different dataset.

### Innovation
We hope to compare the average precision and recall performance between these two models to identify strengths and weakpoints between them. We plan on exploring Pascal VOC, COCO, CityScapes, LVIS, etc. standard datasets and perhaps using more niche datasets such as KITTI to focus on more complex areas of object detection. Another possibility is perhaps obfuscating existing image datasets and see how it affects performance.

### Code
<ol>
  <li>MMDetection Colab - https://colab.research.google.com/drive/1ervLVmxrpWRMnTGFSmjHa9TIRtAr2ag6?usp=sharing</li>
  <li>YOLO Github - https://github.com/rudyorre/object-detection</li>
</ol> 

### Three Relevant Papers
- 'MMDetection: Open MMLab Detection Toolbox and Benchmark' [Code](https://github.com/open-mmlab/mmdetection) [Paper](https://arxiv.org/abs/1906.07155) [1]
- 'SSD: Single Shot MultiBox Detector' [Code](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) [Paper](https://arxiv.org/abs/1512.02325) [2]
- 'Real Time Object/Face Detection Using YOLO-v3' [Code](https://github.com/shayantaherian/Object-detection) [Paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [3]

## Reference
[1] Chen, K., Wang, J., Pang, J., Cao, Y., Xiong, Y., Li, X., Sun, S., Feng, W., Liu, Z., Xu, J., Zhang, Z., Cheng, D., Zhu, C., Cheng, T., Zhao, Q., Li, B., Lu, X., Zhu, R., Wu, Y., … Lin, D. (2019, June 17). MMDetection: Open mmlab detection toolbox and benchmark. arXiv.org. Retrieved January 29, 2023, from https://arxiv.org/abs/1906.07155.
[2] Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.-Y., &amp; Berg, A. C. (2016, December 29). SSD: Single shot multibox detector. arXiv.org. Retrieved January 29, 2023, from https://arxiv.org/abs/1512.02325.
[3] Redmon, J., &amp; Farhadi, A. (n.d.). Yolov3: An Incremental Improvement - pjreddie.com. YOLOv3: An Incremental Improvement. Retrieved January 30, 2023, from https://pjreddie.com/media/files/papers/YOLOv3.pdf.
---
