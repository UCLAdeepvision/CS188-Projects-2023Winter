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

<img src="../assets/images/team24/deepfakeExample.gif" alt="Deepfake Example" width="500">


## Proposal:
The main objective of this project is to experiment with existing advanced machine learning techniques for deepfake detection. Specifically, the project aims to investigate and analyze the current state-of-the-art deepfake detection methods, and evaluate the performance of the developed models using a dataset of deepfake videos. 

## Dataset Used:

### Celeb-DF

Celeb-DF is a dataset used for deepfake forensics. It includes 590 original videos collected from YouTube with subjects of different ages, ethnic groups and genders, and 5639 correspondingDeepFake videos. Unlike most other DeepFake datasets, Celeb-DF contains high visual quality videos that better resemble DeepFake videos circulated on the Internet. 

<img src="../assets/images/team24/celebdfExample.jpg" alt="CelebDF Example" width="600">

### Other notes
We also considered Facebook's Deepfake Detection Challenge (DFDC) dataset, which consists of more than 100,000 videos. It is the largest publicly available datset, but we stuck with Celeb-DF as the time for testing is faster due to the dataset being more than 470 GBs. 


## Architectures:

### ResNet LSTM

The first model we will explore is ResNetLSTM. ResNetLSTM is a combination of two deep learning architectures: Residual Neural Networks (ResNet) and Long Short-Term Memory (LSTM) networks. ResNet is a type of deep neural network that is specifically designed to address the problem of vanishing gradients that can occur in very deep networks. It does this by using residual connections to skip over certain layers in the network, which can help information flow more easily through the network. LSTM networks, on the other hand, are a type of recurrent neural network that are well-suited for processing sequential data, such as time-series data (in our case, video examples). The implementation of ResNetLSTM we will be evaluating consists of a Resnet50 + LSTM with 512 hidden units. The model was pretrained on the ImageNet dataset and will be fintuned on the CelebDF deepfake detection dataset.

<img src="../assets/images/team24/resnetlstmArchitecture.jpg" alt="ResNetLSTM Architecture" width="700" align="middle">

### MesoNet 
MesoNet is a deep learning architecture designed for the detection of manipulated images. MesoNet is based on a combination of convolutional neural networks (CNNs) and long short-term memory (LSTM) networks. However, the specific MesoNet architecture we will be exploring, MesoInception4 (developed by Darius Afchar, Vincent Nozick, Junichi Yamagishi, Isao Echizen) does not implement an LSTM block.  Mesoinception4 is an extension of the Inception architecture, which is a family of deep neural networks that are commonly used for image classification and recognition tasks. It is based on a combination of Inception and Inception-ResNet blocks. These blocks use different types of convolutional layers, including 1x1, 3x3, and 5x5 convolutions, to extract features from the input image. The architecture also includes other notable neural network layers such as max pooling layers, which aggregates the features across the spatial dimensions of the image, and dropout layers to help regularize our model. The model was pretrained on the Mesonet dataset and will be fintuned on the CelebDF deepfake detection dataset.

<img src="../assets/images/team24/mesonetArchitecture.jpg" alt="Mesonet Architecture" width="500" align="middle">

### Other Notes
Othe notable deepfake detection models include EfficientNet B1 LTSM and Xception architectures.


## Training
ResNetLSTM is pretrained on ImageNet, while Mesonet is pretrained on Mesonet. Both models apply transfer learning for deepfake detection. They are finetuned on CelebDF with the following procedure, settings, and results:

### Setup
1. Clone our repository into your environment: https://github.com/jchangz01/CS188-Project-Deepfake-Detection
2. Download `Celeb-DF`(v1) from https://github.com/yuezunli/celeb-deepfakeforensics (NOTE: It is quite large ~2GB)
3. Place `Celeb-DF` directory into `[yourpath]/CS188-Project-Deepfake-Detection/data`

### Training
The following code block shows how we train our models to the CelebDF dataset (exammple is ResNetLSTM specific):
```
import train 

dataset = 'celebdf'
data_path = root + 'CS188-Project-Deepfake-Detection/data/celebdf'
method = 'resnet_lstm_celebdf'
img_size = 224
normalization = 'imagenet'
data = label_data(dataset_path=data_path,
                      dataset=dataset, method=method, 
                      face_crops=True, test_data=False)
augs = df_augmentations(img_size, strength='weak')
folds = 5
epochs = 30
batch_size = 4
lr = 0.0001

model, average_auc, average_ap, average_acc, average_loss = train.train(dataset=dataset, data=data,
                                                                        method=method, img_size=img_size, normalization=normalization, augmentations=augs,
                                                                        folds=folds, epochs=cls.epochs, batch_size=cls.batch_size, lr=cls.lr
                                                                        )
```
                                                                                
### Hyperparameters and Train Results
We used the following hyperparameters and settings to train our models:
- **ResNetLSTM**: image_size = 224, normalization = 'imagenet', folds = 5 (20 val/80 train split) , epochs = 30, batch_size = 4, lr = 0.0001, optimizer: Adam
- **Mesonet**: image_size = 256, normalization = 'xception', folds = 5 (20 val/80 train split) , epochs = 20, batch_size = 32, lr = 0.0001, optimizer: Adam

The results received from training are as follows:
- **ResNetLSTM**: val_loss = 0.1415, val_acc = 0.9457, epochs until best model = 25
- **Mesonet**: val_loss = 0.4458, val_acc = 0.7994, epochs until best model = 20


## Testing

### Procedure
We wanted to observe the difference between 
We observed that further finetuning for Mesonet and ResNetLSTM on the CelebDF dataset did not increase validation accuracy compared to the model weights that were already finetuned to those datasets. Therefore, we used the given model weights rather than the weights we trained, as this gave us higher validation accuracies and reduced the needed computing units and time. To test the finetuned model, we ran the models through the CelebDF test dataset to see the final testing accuracies. The process for this is shown below:

- code block of training one model

### Data Augmentation
As we noticed that further finetuning of the models onto the datasets did not increase accuracies past already given finetuned model weights, we wanted to test the robustness of these models on newer data. To do this, we used Python cv2 and vidaug libraries to perform data augmentation on the test videos, and ran the models through the new data to observe the robustness of the models. We tested with different amounts of augmentation as shown below, intially starting with just rotations and flips, before adding such as gaussian noise, grayscale, dropout, and augmentation factors.

```
from imgaug import augmenters as iaa

seq = iaa.Sequential([
      iaa.Multiply((0.5, 1.5)),
      iaa.GaussianBlur(sigma=(0.0, 3.0)),
      iaa.Dropout((0.0, 0.2)),
      iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
      iaa.ContrastNormalization((0.5, 2.0)),
      iaa.Grayscale(alpha=(0.0, 1.0)),
      iaa.Flipud(flip),
      iaa.Affine(rotate=rotation_angle),
    ])
```

This resulted in three levels of data augmentation on the dataset. We have the original test videos, slightly rotated and/or flipped test videos, and heavily augmented videos. Examples of the three are shown below:

No augmentation:
<img src="../assets/images/team24/mesonetArchitecture.jpg" alt="Mesonet Architecture" width="250" align="middle">

Rotation/flipping:
<img src="../assets/images/team24/mesonetArchitecture.jpg" alt="Mesonet Architecture" width="250" align="middle">

Heavy augmentation (as in code block above):
<img src="../assets/images/team24/mesonetArchitecture.jpg" alt="Mesonet Architecture" width="250" align="middle">

### Results
When testing the finetuned model weights on the CelebDF test dataset, we observed the following metrics. This is on the CelebDF test dataset with no data augmentation:

|          | Mesonet | Mesonet (finetuned) | ResNetLSTM | ResNetLSTM (finetuned) |
| :------- | :------: | -----------------: | ---------: | ---------------------: |
| accuracy |   0   |               0 |       0 |                   0 |
| AUC      |   0   |               0 |       0 |                   0 |
| metric   |   0   |               0 |       0 |                   0 |
| metric   |   0   |               0 |       0 |                   0 |

After applying small rotations and flips to test images in the dataset, we ran the model through the videos again and got the following metrics:

|          | Mesonet | Mesonet (finetuned) | ResNetLSTM | ResNetLSTM (finetuned) |
| :------- | :------: | -----------------: | ---------: | ---------------------: |
| accuracy |   0.257   |               0 |       0 |                   0 |
| AUC      |   0.314   |               0 |       0 |                   0 |
| metric   |   0.   |               0 |       0 |                   0 |
| metric   |   0   |               0 |       0 |                   0 |

Finally, we tried applying large amounts of data augmentations and ran the model through the videos once again to get the following metrics:

|          | Mesonet | Mesonet (finetuned) | ResNetLSTM | ResNetLSTM (finetuned) |
| :------- | :------: | -----------------: | ---------: | ---------------------: |
| Loss     |   0.741   |               16.36 |       0.725 |                   0.948 |
| AUC      |   0.257   |               0.356 |       0.495 |                   0.465 |
| AP       |   0.314   |               0.356 |       0.495 |                   0.465 |
| accuracy  |  0.314   |               0.257 |       0.257 |                   0.486 |

## Conclusion

- initially, resnetlstm does better than mesonet. a hypothesis for this is that it is able to learn temporal dependecies and can relate frames in a video together.
- applying data augmentation shows the model is extremely sensitive to new distributions, wasn't able to hold as well as we would've expected.
- definitely lots more work to be done, deepfakes=bad and we want to change that

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
