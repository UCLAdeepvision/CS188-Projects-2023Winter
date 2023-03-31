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

## Data

This project utilizes a dataset of historical photos. It is divided into source and target images where the target image quality is much lower than that of the source image quality. Therefore, this dataset is the optimal use case of image matching with deep neural networks because for a dataset whose source and target are of the same image quality, methods such as Image hashing would be faster and deliver excellent results. 

Below are the source and target images:

[SourceImages](https://drive.google.com/drive/folders/15FyEpM2U4v5yT4N-daAyPZvns759nhmh?usp=sharing)
[TargetImages](https://drive.google.com/drive/folders/1YGVlKhAnx5kjjvACAwbWHvFt5OU6eHd6?usp=sharing)

## Data Preprocessing

During this process we resize every image to 224 by 224. Then a data augmentation applied on each of the images. This augmentation includes a random color jitter, random cropping, and random horizontal flip. These motivation behind using these specific augments are mainly to minimize the difference between the source images and the target images since the target images usually have a different hue, color,  saturation and contrast. Target images can also be flipped, look cropped, and even rotated and more blury compared to the source image. To keep things simple I stuck with those three augments so that the model will be more robust to different hue, color, saturation, contrast, cropping, and horizontal orientation when outputing feature vectores for comparison.
 

## Training

The primary method of comparing images is through the cosine similarity of the feature vector outputed by the model for each image input. Therefore the goal of training is to improve the feature vectors. We can improve the feature vectors by training on matching pairs and minimizing the differences between their vectors. If we can inject random noise into one of the images, we can make the model find the correct matches even if the images are slightly different (angled, brighter, cropped, etc). This can be done by applying data augmentations to the souce image which would be used to create the pairings. Additionally, we will be training on pairs where there will be positive and negative pairs in order for the model to differentiate between a matching pairs and non-matching pairs. Using this method does not require any labeling and can be trained with a large number of images with ease.

In our case, we can use a pretrained model and freeze those layers and add a couple layers below that will be trained on. This has the benefit of less gpu utilization meaning that we can train with larget batches which leads to faster training. In addition, because of our small dataset, it is ideal to train less parameters, so freezing the pretrained model would make sense since that would require a lot of training data. Also, without changing the weights of the pretrained model, we can reduce the chances that the entire model will overfit our data, allowing for better generalization.

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

This model uses one of the pretrained model from https://pytorch.org/vision/stable/models.html and attached under it are two fully connected layers that project the output of the pretrained model to a 1000 dimension vector to represent each image.

For training, we will use Normalized temperature-scaled cross entropy loss. Because we are training in batches, we will have batch size/2 pairs. When calculating the loss for a single pair, we treat all other images inside the batch as the negative pairing. We find the cosine similarity of the source image with respect to all other images and essentially use cross entropy loss with the cosine similarity of the positive pair as the "label".

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

## Matching

In order to match images, we perform cosine similarity on each of the vector representation of the source images and pair it with every one of the vector representations of the target images. From there, we take the top N cosine similarity scores to ideally get the top N most similar images.

If you want to utilize the automatic scoring function for matching on a custom data set, you must make sure that the image in the source folder along with its match in the target folder are named exactly the same.

## Results

The metric used to compare the vector representation of the images is cosine similarity since that is what we are trying to minimize in our contrastive loss function for positive pairs. The evaluation metric used overall to assess the model is Top-N Accuracy meaning that for a single input image, if one of the top-N contains the correct image match than the image is counted as correctly matched.

For this method, I tested vit_b_16 (162/230), efficientnet_v2_m (80/230), swin_t (169/230), resnet50 (144/230), vgg16 (103/230), convnext_small (163/230), and regnet_x_32gf (127/230). I trained on 15 epochs and for top 15 accuracy.

## Ensemble

In addition, we can use a combination of pretrained models to form an ensemble by concatenating the individual outputs of each model. Based on the number of models we combine, we can easily adjust the number of input and output features of our fully connected layers since the pretrained models on pytorch output a 1000 dimension vector per image.

Considering that the pretrain models on pytorch output a 1000 dimension vector, we would use a stack of layers underneath in order to reduce the dimension back to 1000. When training the ensemble, we will also freeze the layers of the pretrained models and only be training the linear layers underneath. Ensemble models will be able to increase the accuracy compared to the single model. By testing different pretrained models with the SingleModel, we may be able to combine the best models in order to create an even better performing model.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class Ensemble(nn.Module):
    def __init__(self):
        super(Ensemble, self).__init__()
        self.model1 = models.vit_b_16(weights='IMAGENET1K_SWAG_LINEAR_V1')
        self.model2 = models.convnext_small(weights='IMAGENET1K_V1')
        self.model3 = models.swin_t(weights='DEFAULT')
        num_features = 1000
        self.fc1 = nn.Linear(num_features*3, num_features*2)
        self.fc2 = nn.Linear(num_features*2, num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.model1(x.clone())
        x2 = self.model2(x.clone())
        x3 = self.model3(x.clone())


        x = torch.cat((x1, x2, x3), dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

```

### Results

- Top 1 Acc: 110/230
- Top 2 Acc: 138/230
- Top 3 Acc: 146/230
- Top 4 Acc: 158/230
- Top 5 Acc: 160/230
- Top 6 Acc: 170/230
- Top 7 Acc: 171/230
- Top 8 Acc: 172/230
- Top 9 Acc: 173/230
- Top 10 Acc: 175/230
- Top 11 Acc: 176/230
- Top 12 Acc: 178/230
- Top 13 Acc: 179/230
- Top 14 Acc: 183/230
- Top 15 Acc: 184/230

## Discussion

Overall, on a single model basis, the pretrained models that performed the best were convnext_small, vit_16, swin_t. These three models were the highest accuracy among the models I have chosen based on the ACC@1 and ACC@5 scores on the pytorch website and the "newer" models based on transformers. This makes sense because we are always freezing the pretrained models, that means we are only training the projection layers (two fully connected layers). Because we are essentially only using two layers to improve the vector representation, it becomes very important to use a good base model such as the transformer based models.

Additionally, using the largest batch size possible was very important in terms of speeding up the training process, since this image matching method is supposed to be trained in order to be tailored to a specific dataset meaning we would need to re-train the model again for a different dataset. This is why the fact that the learning method is unsupervised is very important because it allows any user to use different datasets instantly without any labeling, and the large batch size would make this process faster.

Using the three best models, I combined them into an ensemble and combined their outputs by modifing the projection layer to accept a 3000 dimension vector and finally output of 1000 dimension vector at the final fully connected layer. This test increased the accuracy by about 6.5%. This means that there were many images that could not be correctly matched by any of the models I used.

When compared to methods that do not involve deep learning, the deep learning method perform much better, but slower. ImageHashing specifically, scored only around 30% (~69/230). This shows this contrastive training using augmented images helped generalize the vector representations to be more robust to the major drop in image quality between the source and target images.

Looking at the incorrect images specifically confirms that many are difficult to match due to the large quality differences. If I had more time, ways that could alleviate this could be using different augmentations. Another more interesting project would be using image generation to generate a similar image of a source image into the same image but in the quality of the target image and perform image hashing or use this current model to match the images.

The maximum accuracy for matching ultimately will depend on the dataset itself and any differences in quality between the source and target images: contrast, hue, brightness, saturation, flip, rotations, etc. However, using deep learning and data augmentation methods, we are able to minimize the effect that those differences have in matching images.

## Code and Video

[Code](https://drive.google.com/drive/folders/1jeZJRZK4qVRYOcr9dVvxYDDoHfBgGjP5?usp=share_link)

[Video](https://youtu.be/XtV3CYQ4-wI)

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
