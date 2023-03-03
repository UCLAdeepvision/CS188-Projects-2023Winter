---
layout: post
comments: true
title: Exploring CLIP
author: Tang Mohan
date: 2023-01-29
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

# Week 3: Project Proposal
## Description
The goal of this project is to explore and understand CLIP - Cross-Lingual Image-Text Pre-training. Previous Methods for computer vision typically requires large amount of data and are trained only to perform well in one specific task. CLIP is able to solve these problems by jointly training image encoder and text encoder. It is able to train a model that can move to a task that it has not seen before without large amount of extra data.  For our project, we want to run the code for CLIP on our own and explore its properties. 

## 3 Relevant Papers and Code Repositories
[1] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G. & Sutskever, I.. (2021). Learning Transferable Visual Models From Natural Language Supervision. *Proceedings of the 38th International Conference on Machine Learning*, in *Proceedings of Machine Learning Research* 139:8748-8763 Available from https://proceedings.mlr.press/v139/radford21a.html.

Code Repository: https://github.com/openai/CLIP

[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In *Advances in neural information processing systems*, pp. 5998–6008, 2017.

Code Repository: https://github.com/tensorflow/tensor2tensor

[3] He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016b.

Code Repository: https://github.com/KaimingHe/deep-residual-networks

# Week 7: Midterm Report

## How CLIP Works and the Algorithm
CLIP is a method to pretrain a model. CLIP trains a network that extracts features from an image, whose results can be used for later layers to do image classification. 

CLIP trains an image encoder and a text encoder. "Encoders" mean networks that map the input image to a vector. The goal is to use the image encoder and the text encoder to map the images and texts to the same space -- so that the same vector can represent key features from both images and texts. 

The training process is contrasive. The image encoder and the text encoder are trained together to minimize the distance between image_encoder(image) and text_encoder(text) of a correct (image, text) pair in the dataset. The actual loss function captures this metric. The loss function has two parts, calculating cross entropy loss from two directions. The first part calculates distances from each encoded image to all the encoded texts, and calculate the cross entropy loss between such distances and the correct text associated with this image. The second part calculates distances from each encoded text to all the encoded images, and calculate the cross entropy loss between such distances and the correct image associated with this text. The actual loss is the average of the two. 

The algorithm is as follows: [1]

```
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter
# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]
# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)
# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)
# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
```

There are models in computer vision and natural language processing that can be used as encoders. The image encoder is based on ResNet and the text encoder is based on Transformer. 

## Run the example code
To begin, we run the example code at https://github.com/openai/CLIP. 
We first configure the environment by running
```
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

Then, using the example code we train a logistic regression model above the pretrained CLIP model to do image classification. We only update the parameters of the logistic model, without fine tuning the parameters of the encoders. The code is as follows:
```
import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Load the dataset
root = os.path.expanduser("~/.cache")
train = CIFAR100(root, download=True, train=True, transform=preprocess)
test = CIFAR100(root, download=True, train=False, transform=preprocess)


def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")
```
We are able to get the following result:
![YOLO]({{ '/assets/images/38/example result.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

The accuracy is 79.920. This is better than ResNet-1001 on CIFAR 100, which is the dataset used. As we know, the linear regression model is a very simple model, so this is showing the "One-Shot" property of CLIP -- trained on one dataset, it is able to generalize to a different context with small amount of additional training. 

## Our Plan for Modification
It is possible that an image can be properly matched to multiple texts. However, in the training process of CLIP, the algorith try to make it such that only the distance to the exact label of the image should be maximized, and the distance to all the other texts should be minimized. If an image is assigned to text T1, and encoder(T1) and encoder(T2) are similar, the algorithm will still want the encoded image to be far from encoder(T2), disregarding the fact that the encoded value of T1 and T2 are similar. We want to adjust this and take the loss function to be expected distance between the encoded value of the text and the encoded value of the label, where the distance from the the encoded value of the text to the encoded image should be considered as the probability in the calculation of the expectation. This is still a rough idea. We need to modify this a bit to make sure that the algorithm don't just find the trivial solution by mapping everything to the same vector. 

## Main Content
Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)



## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.


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
