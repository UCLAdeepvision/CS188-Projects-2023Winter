---
layout: post
comments: true
title: Object Detection
author: Ryan Vuong and Travis Graening
date: 2023-02-25
---


> In this project, we wish to dive deep and examine the YOLO v7 (You Only Look Once) object detection model while exploring any possible improvements. YOLO v7 is one of the fastest object detection models in the world today which provides highly accurate real-time object detection. During our examination, we will inspect the layers and algorithms within YOLO v7, test the pre-trained YOLO v7 model provided by the YOLO team, and train and test YOLO v7 against different datasets.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## 1. Introduction
Put stuff here later

## 2. Setting Up YOLO v7

### 2.1 Loading Pre-Trained Model
Code based off: [Colab Notebook](https://colab.research.google.com/gist/AlexeyAB/b769f5795e65fdab80086f6cb7940dae/yolov7detection.ipynb)
<br>
Within the Official YOLOv7 repository, there are a few pre-trained YOLOv7 models that have been trained solely through the MS COCO Dataset. Some of these models are designed to work better on edge devices, such as YOLOv7-tiny, while others are designed to work better on stronger cloud devices, such as YOLOv7-W6. For the purposes of this demo, I will use the standard YOLOv7 pre-trained model.

1. Basic Setup: import sys, torch and clone the YOLOv7 repository.
```
import sys
import torch
!git clone https://github.com/WongKinYiu/yolov7
```

2. Download the pre-trained model:
```
!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

3. Run object detection:
```
!python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/yolo-test1.jpg
```

4. Define helper function to view images and load image:

```
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

imShow("runs/detect/exp5/yolo-test1.jpg")

```
![detected image of friends on a hike](/CS188-Projects-2023Winter/assets/images/team31/yolo-test1.jpg)

[comment]: <> (<img src="/CS188-Projects-2023Winter/assets/images/team31/yolo-test1.jpg" width="640" height="640" />)
<br>


## References and Code Bases

**YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors**

Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao
- [Paper](https://arxiv.org/pdf/2207.02696.pdf)
- [Code](https://github.com/WongKinYiu/yolov7)

**Multiple Object Recognition with Visual Attention**

Jimmy Ba, Volodymyr Mnih, Koray Kavukcuoglu
- [Paper](https://paperswithcode.com/paper/multiple-object-recognition-with-visual)
- [Code](https://paperswithcode.com/paper/multiple-object-recognition-with-visual#code)

**Scalable Object Detection Using Neural Networks**

Dumitru Erhan, Christian Szegedy, Alexander Toshev, Dragomir Anguelov
- [Paper](https://paperswithcode.com/paper/scalable-object-detection-using-deep-neural)
- [Code](https://paperswithcode.com/paper/scalable-object-detection-using-deep-neural#code)

**Objects as Points**

Xingyi Zhou, Dequan Wang, Philipp Krähenbühl
- [Paper](https://www.semanticscholar.org/paper/Objects-as-Points-Zhou-Wang/6a2e2fd1b5bb11224daef98b3fb6d029f68a73f2)
- [Code](https://github.com/xingyizhou/CenterNet)

**Pointly-Supervised Instance Segmentation**

Bowen Cheng, Omkar Parkhi, Alexander Kirillov: Pointly-Supervised Instance Segmentation
- [Paper](https://arxiv.org/pdf/2104.06404.pdf)
- [Code](https://github.com/facebookresearch/detectron2/tree/main/projects/PointSup)

**End-to-End Object Detection with Transformers**

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko
- [Paper](https://arxiv.org/pdf/2005.12872.pdf)
- [Code](https://github.com/facebookresearch/detr#detr-end-to-end-object-detection-with-transformers)

## Basic Syntax
### Image


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

---