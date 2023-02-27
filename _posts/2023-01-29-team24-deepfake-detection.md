---
layout: post
comments: true
title: Deepfake Detection
author: Justin Kyle Chang, Oliver De Visser
date: 2022-01-29
---

> Detecting synthetic media has been an ongoing concern over the recent years due to the increasing amount of deepfakes on the internet. In this project, we will explore the different methods and algorithms that are used in deepfake detection.

<!--more-->

{: class="table-of-content"}

-   TOC
    {:toc}
    
## Introduction: 
Deepfakes, or artificial intelligence-generated videos that depict real people doing and saying things they never did, have become a growing concern in recent years. These artificially generated content can be used to spread misinformation, manipulate public opinion, and even harm individuals. Therefore, the ability to detect deepfakes is crucial to ensure the integrity of information and protect people from potential harm.

## Proposal:
The main objective of this project is to develop and evaluate advanced machine learning techniques for deepfake detection. Specifically, the project aims to investigate and analyze the current state-of-the-art deepfake detection methods, and evaluate the performance of the developed models using a dataset of deepfake videos. 

## Datasets:
### Deepfake Detection Challenge (DFDC) 
The DFDC (Deepfake Detection Challenge) is a Facebook developed dataset for deepface detection consisting of more than 100,000 videos. It is currently the largest publicly available dataset and was created for a competition aimed towards creating new and better models to detect manipulated media. The dataset consists of a preview dataset with 5k videos featuring two facial modification algorithms and a full dataset with 124k videos featuring 8 facial modification algorithms. 


### Celeb-DF
Celeb-DF is a dataset used for deepfake forensics. It includes 590 original videos collected from YouTube with subjects of different ages, ethnic groups and genders, and 5639 correspondingDeepFake videos. Unlike most other DeepFake datasets, Celeb-DF contains high visual quality videos that better resemble DeepFake videos circulated on the Internet. 


## Potential Architectures :
### ResNet LTSM
This is the architecture we will be using for our model.
Implementation of a Resnet50 + LSTM with 512 hidden units as it was described in the paper
DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection 
### EfficientNet B1 LTSM
This is an implementation Efficient Net that was implemented for the DeepFake Detection Model
“To make it comparable with ResNet50 + LSTM it uses the same fully connected layers and also uses 512 hidden units as it was described in the paper”
DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection

### MesoNet 
The MesoInception4 deepfake detection architecture as introduced in MesoNet: a Compact Facial Video Forgery Detection Network  from Darius Afchar, Vincent Nozick, Junichi Yamagishi, Isao Echizen

### XCeption
Creates an Xception Model as defined in:
Francois Chollet, Xception: Deep Learning with Depthwise Separable Convolutions


## ResNet LSTM Implementation:
### data augmentation methods
### optimal hyperparameters
### code examples

## Results:

## Conclusion:

## Demo:
- https://github.com/jchangz01/CS188-Project-Deepfake-Detection
- Currently based off implementations from research papers, not complete and not ready for training yet


## Related Works

-   Combining EfficientNet and Vision Transformers for Video Deepfake Detection
    -   [Paper](https://arxiv.org/abs/2107.02612)
    -   [Github](https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection)
-   Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization
    -   [Paper](https://arxiv.org/abs/2204.06228v1)
    -   [Github](https://github.com/ControlNet/LAV-DF)
-   Video Face Manipulation Detection Through Ensemble of CNNs
    -   [Paper](https://arxiv.org/abs/2004.07676v1)
    -   [Github](https://github.com/polimi-ispl/icpr2020dfdc)
-   DeepFake Detector Performance Model 
    -   [Github] https://github.com/CatoGit/Comparing-the-Performance-of-Deepfake-Detection-Methods-on-Benchmark-Datasets/blob/master/deepfake_detector/pretrained_mods/efficientnetb1lstm.py
-   Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics
    -   [Paper] https://arxiv.org/abs/1909.12962
    -   [Github] https://github.com/yuezunli/celeb-deepfakeforensics
-   DFDC Dataset
    -   [Link] https://ai.facebook.com/datasets/dfdc/
-   Xception: Deep Learning with Depthwise Separable Convolutions
    -   [Paper] https://arxiv.org/pdf/1610.02357.pdf
-   MesoNet: a Compact Facial Video Forgery Detection Network
    -   [Paper] (https://arxiv.org/abs/1809.00888)

## References

[1] Coccomini, Davide, et al. “Combining EfficientNet and Vision Transformers for Video Deepfake Detection.” ISTI-CNR, via G. Moruzzi 1, 56124, Pisa, Italy, 2022.

[2] Cai, Zhixi, et al. “Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization.” Monash University, 2022.

[3] Bonettini, Nicolo, et al. “Video Face Manipulation Detection Through Ensemble of CNNs.” Polytechnic University of Milan, 2020.




##
##
##
## previous blog resources

## Main Content

Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

## Basic Syntax

### Image

Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
_Fig 1. YOLO: An object detection method in computer vision_ [1].

Please cite the image if it is taken from other people's work.

### Table

Here is an example for creating tables, including alignment syntax.

|      | column 1 | column 2 |
| :--- | :------: | -------: |
| row1 |   Text   |     Text |
| row2 |   Text   |     Text |

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

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2016.

---
