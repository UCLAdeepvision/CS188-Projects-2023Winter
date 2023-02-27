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

# Introduction

# Deep Fake Generation?

# Setup and Data Preparation
## Deep Face Lab

## Modified VGG16 CNN (Faceswap)
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



# Model Structure Comparison

# DeepFaceLab & Implementation

# FaceSwap & Implementation 

# Quantitative Results Comparison

# Conclusion

# Video Demo

# Colab Demo

# Papers
[First Order Motion Model for Image Animation](https://papers.nips.cc/paper/2019/hash/31c0b36aef265d9221af80872ceb62f9-Abstract.html) - [repo](https://github.com/AliaksandrSiarohin/first-order-model)

[Fast Face-swap Using Convolutional Neural Networks](https://arxiv.org/abs/1611.09577) - [repo](https://github.com/deepfakes/faceswap#overview)

[DeepFaceLab: A simple, flexible and extensible face swapping framework](https://arxiv.org/pdf/2005.05535v4.pdf) - [repo](https://github.com/iperov/DeepFaceLab)


