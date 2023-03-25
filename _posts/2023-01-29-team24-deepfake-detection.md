---
layout: post
comments: true
title: Deepfake Detection
author: Justin Kyle Chang, Oliver De Visser
date: 2022-01-29
---

> Detecting synthetic media has been an ongoing concern over the recent years due to the increasing amount of deepfakes on the internet. In this project, we will explore the different methods and algorithms that are used in deepfake detection.
    
## Introduction: 
Deepfakes, or artificial intelligence-generated videos that depict real people doing and saying things they never did, have become a growing concern in recent years. These artificially generated content can be used to spread misinformation, manipulate public opinion, and even harm individuals. Therefore, the ability to detect deepfakes is crucial to ensure the integrity of information and protect people from potential harm.


![SampleDeepfake]({{ '/assets/images/team01/deepfakeSampleImage.jpg' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 1. Sample deepfake original (left) and deepfake (right)* [1].


## Proposal:
The main objective of this project is to develop and evaluate advanced machine learning techniques for deepfake detection. Specifically, the project aims to investigate and analyze the current state-of-the-art deepfake detection methods, and evaluate the performance of the developed models using a dataset of deepfake videos. 

## Dataset Used:

### Celeb-DF

Celeb-DF is a dataset used for deepfake forensics. It includes 590 original videos collected from YouTube with subjects of different ages, ethnic groups and genders, and 5639 correspondingDeepFake videos. Unlike most other DeepFake datasets, Celeb-DF contains high visual quality videos that better resemble DeepFake videos circulated on the Internet. 

[Deepfake Example]({{ '/assets/images/team24/Celeb-DF-0000004265-9ebb7ff0.jpg' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
_Fig 1. Celeb-DF example input images from deepfake videos _ [1].



### Other notes
We also considered Facebook's Deepfake Detection Challenge (DFDC) dataset, which consists of more than 100,000 videos. It is the largest publicly available datset, but we stuck with Celeb-DF as the time for testing is faster due to the dataset being more than 470 GBs, 


## Architectures :

### ResNet LSTM

This is the architecture we will be using for our model.
Implementation of a Resnet50 + LSTM with 512 hidden units as it was described in the paper
DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection 

[ResNet LSTM]({{ '/assets/images/team24/ResNet-LSTM-model-The-signal-is-first-fed-into-ResNet-as-a-three-channel-input-phasic.jpg' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
_Fig 1. Uses LSTM blocks after applying ResNet architectures _ [1].


### MesoNet 
The MesoInception4 deepfake detection architecture as introduced in MesoNet: a Compact Facial Video Forgery Detection Network  from Darius Afchar, Vincent Nozick, Junichi Yamagishi, Isao Echizen

[ResNet LSTM]({{ '/assets/images/team24/Screen Shot 2023-03-17 at 4.44.26 PM.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
_Fig 1. MesoNet architecture _ [1].

### Other Notes
We also considered using and testing the EfficientNet B1 LTSM and Xception architectures


## Training
 - didn't actual train as too computationally expensive, evaluated pretrained weights
 - would do this by ...

### Hyperparameters
- hyperparameters used

### Fine-tuning
- models are trained and then fine-tuned on our datasets

## Testing

### Procedure
- example image from code
- code segments


### Results
- tables with our accuracies

## Conclusion
- observations

## Demo
- https://github.com/jchangz01/CS188-Project-Deepfake-Detection



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
-   Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics
    -   [Paper](https://arxiv.org/abs/1909.12962)
    -   [Github](https://github.com/yuezunli/celeb-deepfakeforensics)
-   MesoNet: a Compact Facial Video Forgery Detection Network
    -   [Paper](https://arxiv.org/abs/1809.00888)
-   DeepFake Detector Performance Model 
    -   [Github](https://github.com/CatoGit/Comparing-the-Performance-of-Deepfake-Detection-Methods-on-Benchmark-Datasets/blob/master/deepfake_detector)


## References

[1] Coccomini, Davide, et al. “Combining EfficientNet and Vision Transformers for Video Deepfake Detection.” ISTI-CNR, via G. Moruzzi 1, 56124, Pisa, Italy, 2022.

[2] Cai, Zhixi, et al. “Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization.” Monash University, 2022.

[3] Bonettini, Nicolo, et al. “Video Face Manipulation Detection Through Ensemble of CNNs.” Polytechnic University of Milan, 2020.

[4] Li, Yuezun, et al. “Celeb-DF: A Large-Scale Challenging Dataset for Deepfake Forensics.” ArXiv.org, 16 Mar. 2020, 






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
