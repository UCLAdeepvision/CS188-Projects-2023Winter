---
layout: post
comments: true
title: Image Similarity
author: Brandon Le
date: 2023-01-29
---


> Image Similarity has important applications in areas such as retrieval, classification, change detection, quality evaluation and registration. Here, we aim to improve the performance of models to pair images based on their similarities.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Project Proposal

The goal of this project is to understand and test training methods that could increase the effectiveness of models to identify the similarity between pairs of images.


## Methods of Training

### Unsupervised Methods

The primary method of comparing images is through the cosine similarity of the feature vector outputed by the model for each image input. Therefore the goal of training is to improve the feature vectors. We can improve the feature vectors by training on matching pairs and minimizing the differences between their vectors. If we can inject random noise into one of the images, we can make the model find the correct matches even if the images are slightly different (angled, brighter, cropped, etc). This can be done by applying data augmentations to the souce image which would be used to create the pairings. Additionally, we will be training on pairs where there will be positive and negative pairs in order for the model to differentiate between a matching pairs and non-matching pairs. Using this method does not require any labeling and can be trained with a large number of images with ease.

In our case, we can use a pretrained model and freeze those layers and add a couple layers below that will be trained on. This has the benefit of faster training and requires a lot less training images in order to fine tune. Also, without changing the weights of the pretrained model, we can reduce the chances that the entire model will overfit our data, allowing for better generalization.

Below is the class for the model we will be using.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class SingleModel(nn.Module):
    def __init__(self):
        super(SingleModel, self).__init__()
        self.model = models.vgg16(weights='DEFAULT')

        num_features = 1000
        self.fc1 = nn.Linear(num_features, num_features)
        self.fc2 = nn.Linear(num_features, num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.model(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

```

We will also try training without freezing the model in order to test the effects of contrastive learning on smaller training sets.

In addition, we can use a combination of pretrained models to form an ensemble by concatenating the individual outputs of each model. Considering that the pretrain models on pytorch output a 1000 dimension vector, we would use a stack of layers underneath in order to reduce the dimension back to 1000. When training the ensemble, we will also freeze the layers of the pretrained models and only be training the linear layers underneath. Ensemble models will be able to increase the accuracy compared to the single model. By testing different pretrained models with the SingleModel, we may be able to combine the best models in order to create an even better performing model.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class Ensemble(nn.Module):
    def __init__(self):
        super(Ensemble, self).__init__()
        self.model1 = models.vgg16(weights='DEFAULT')
        self.model2 = models.resnet50(weights='IMAGENET1K_V2')
        self.model3 = models.inception_v3(weights='IMAGENET1K_V1')
        num_features = 1000
        self.fc1 = nn.Linear(num_features*3, num_features*2)
        self.fc2 = nn.Linear(num_features*2, num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.model1(x.clone())
        x2 = self.model2(x.clone())
        x3 = self.model3(x.clone())

        if self.training:
            x = torch.cat((x1, x2, x3.logits), dim=1)
        else:
            x = torch.cat((x1, x2, x3), dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

```

For training, we will use Normalized temperature-scaled cross entropy loss mentioned in the [SimCLR](https://arxiv.org/pdf/2002.05709.pdf) paper. Because we are training in batches, we will have batch size/2 pairs. When calculating the loss for a single pair, we treat all other images inside the batch as the negative pairing. We find the cosine similarity of the source image with respect to all other images and essentially use cross entropy loss with the cosine similarity of the positive pair as the "label".

Below is our implementation of the custom loss function. In this implementation we use target in order to index the positive matching image for each training example. The target parameter does not represent labels, but its elements represent indexes to the matching image. The elements of target should coincide with how you decide to order the source images and the augmented images within the batch.

```python
def contrast_loss_func(output, target, temp=0.05):
    norm_out = F.normalize(output, dim=1)
    similarity = torch.matmul(norm_out, norm_out.T)/temp
    
    similarity = similarity.fill_diagonal_(-float('inf'))
    neg = torch.sum(torch.exp(similarity), dim=-1)

    N = similarity.shape[0]
    pos = torch.exp(similarity[torch.arange(N), target])
    loss = -torch.log(pos/neg).mean()

    return loss
```

### Supervised Method

Another method of training involves supervised learning. If we have a small amount of images, we can apply a label for each unique image as 1, 2, ..., n. We can then train for classification on the images we wish to find matches within a separate pool. For this method we can simply use cross entropy loss instead of the custom loss function above since we are training with labels. Then, on testing we simply output a class for each image input to find the exact matches.

Below we made a small edit to the original SingleModel class where the final connected layer outputs a vector with a dimension equal to the number of images we want to find matches to. We can do the same for the ensemble model as well.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class SingleModel(nn.Module):
    def __init__(self, num_img):
        super(SingleModel, self).__init__()
        self.model = models.vgg16(weights='DEFAULT')

        num_features = 1000
        self.fc1 = nn.Linear(num_features, num_features)
        self.fc2 = nn.Linear(num_features, num_img)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.model(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

```
In this experiment we will be able to compare the performance of the supervised method with the unsupervised method in finding perfect matches.


## Relevant Research Papers

**A Simple Framework for Contrastive Learning of Visual Representations** 
- [Paper](https://arxiv.org/pdf/2002.05709.pdf)
- [Code](https://github.com/google-research/simclr)

**Identical Image Retrieval using Deep Learning** 
- [Paper](https://arxiv.org/pdf/2205.04883.pdf)
- [Code](https://github.com/sayannath/Identical-Image-Retrieval)

**Self-supervised PRoduct Quantization for Deep Unsupervised Image Retrieval** 
- [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jang_Self-Supervised_Product_Quantization_for_Deep_Unsupervised_Image_Retrieval_ICCV_2021_paper.pdf)
- [Code](https://github.com/youngkyunJang/SPQ) 

**Image Similarity using Deep CNN and Curriculum Learning** 
- [Paper](https://arxiv.org/ftp/arxiv/papers/1709/1709.08761.pdf)
