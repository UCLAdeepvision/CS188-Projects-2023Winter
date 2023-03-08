---
layout: post
comments: true
title: Deepfake Generation
author: Freddy Aguilar, Luis Frias
date: 2022-01-30
---

> Deepfake Generation
<!--more-->
{: class="table-of-content"}
* TOC
{:toc}


> We chose to explore the topic of [deepfake generation](https://github.com/aerophile/awesome-deepfakes). 

## Introduction
Recent years have seen a tremendous advancement in deep fake technology, with new models and algorithms emerging that enable more convincing and realistic picture swaps. There are many uses for the skill of seamlessly and convincingly swapping out a person's face, from entertainment to political influence. But as deep fake technology spreads, so is the need to comprehend its strengths and weaknesses. The first model is based on Generative Adversarial Networks (GANs), and the second model is based on a modified version of the VGG16 CNN architecture. In this article, we will compare and contrast these two deep fake generation methods for picture swap. In order to learn more, we will evaluate each model's performance and take into account its advantages and disadvantages. We intend to provide light on the current state of deep fake creation for picture swap and add to ongoing discussions regarding its appropriate use by comparing these models side by side.

## Deep Fake Generation?
Deep fake generation is the process of producing synthetic media that closely resembles the appearance and behavior of actual people is known as "deep fake generation. This technology can produce incredibly lifelike images, films, and audio. For the intention of limiting the scope of the article, we will focus on pictures swaps. Face swapping is a specific type of deep fake generation that involves replacing one person's face with another person's face in an image or video. 

Face swapping typically entails feeding the network pairs of photos, each of which has the face of one person and the face of another. The network then learns to recognize the features and characteristics of each person's face and the relationship between them.

Once trained, the face swap model can be used to create new pictures or movies with the faces of two persons switched. It is feasible to create highly realistic face swaps that are challenging to discern from actual photographs by carefully tweaking the network's parameters after training it on a sizable collection of facial images.


## Setup and Data Preparation

## Model Structure Comparison

## DeepFaceLab & Implementation

## FaceSwap & Implementation 

Modified VGG16 CNN 

The folowing [colab](https://colab.research.google.com/drive/1_j_0N9uCR47ms5paXTKgQVTq1M4GX-9M?usp=sharing) will be used as the setup but there is also a [locally based](https://github.com/deepfakes/faceswap/blob/master/INSTALL.md) installation method based on your hardware configuration.

The process can be divided into three parts:

1. Extract
    
    This step will take photos from an `src` folder and extract faces into an `extract` folder.

    ```
    from google.colab import files
    import os
    import zipfile
    !rm -rf src
    src_path = 'src'
    os.mkdir(src_path)
    # Upload zip file(s) named after person/face containing face images of source (Ex. obama.zip)
    uploaded = files.upload()
    for k in uploaded.keys():
      zip_file = zipfile.ZipFile(k, 'r')
      zip_file.extractall(src_path)
      print("Uploaded and unzipped:", k)

    !rm -rf faces
    face_path = 'faces'
    os.mkdir(face_path)
    for face in os.listdir(src_path):
        !python faceswap.py extract -i src/"{face}" -o faces/"{face}"
    ```
    The following snippet takes in a zip file from user input where the zip would contain images of a person's face (best to use multiple angles for better results). The zip should be named after the person names so the files should be output at `src/{name}`. Then for every face input, the registered images would be output in the `face/{name}` folder.


2. Train

    This step will take photos from two folders containing pictures of both faces and train a model that will be saved inside the `models` folder.

    ```
    # Once multiple faces registers
    a = input("First face name to extract:")
    b = input("Second name to extract:")
    !python faceswap.py train -A faces/'{a}' -B faces/'{b}' -m models/"{a}_to_{b}_model"
    ```
    The following snippet takes two inputs where you would input the names of faces you want to replace where first face is the source and seconds face is the destination. Make sure to have atleast 25 images for each face.

3. Convert

    This step will take photos from original folder and apply new faces into `modified` folder.

    ```
    # Once we finished training,
    # we can convert faces now
    a = input("Name to source from(replace): ")
    b = input("Name of face origin: ")
    !python faceswap.py convert -i src/'{a}' -o converted/ -m models/"{a}_to_{b}_model"
    ```
    The following snippet takes the input of what face you want replaced and then asks which face you want to use to replace. If a trained model exists it will proceed and output in the `converted` folder


## Quantitative Results Comparison

## Conclusion

## Video Demo

## Colab Demo

[First Order Motion Model for Image Animation](https://papers.nips.cc/paper/2019/hash/31c0b36aef265d9221af80872ceb62f9-Abstract.html) - [repo](https://github.com/AliaksandrSiarohin/first-order-model)

[Fast Face-swap Using Convolutional Neural Networks](https://arxiv.org/abs/1611.09577) - [repo](https://github.com/deepfakes/faceswap#overview)

[DeepFaceLab: A simple, flexible and extensible face swapping framework](https://arxiv.org/pdf/2005.05535v4.pdf) - [repo](https://github.com/iperov/DeepFaceLab)


