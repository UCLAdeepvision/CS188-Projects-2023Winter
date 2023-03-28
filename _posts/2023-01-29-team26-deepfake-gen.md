---
layout: post
comments: true
title: Deepfake Generation
author: Freddy Aguilar, Luis Frias
date: 2022-01-30
---

> This post explores DeepFake Generation. In this artcle, we explore two models : GAN and CNN. The report analyze's the similarties and differences of the two models when conducting Faceswap. 

> Deepfake Generation
<!--more-->
{: class="table-of-content"}
* TOC
{:toc}


# Introduction
Recent years have seen a tremendous advancement in deep fake technology, with new models and algorithms emerging that enable more convincing and realistic picture swaps. There are many uses for the skill of seamlessly and convincingly swapping out a person's face, from entertainment to political influence. But as deep fake technology spreads, so is the need to comprehend its strengths and weaknesses. The first model is based on Generative Adversarial Networks (GANs), and the second model is based on a modified version of the VGG16 CNN architecture. In this article, we will compare and contrast these two deep fake generation methods for picture swap. In order to learn more, we will evaluate each model's performance and take into account its advantages and disadvantages. We intend to provide light on the current state of deep fake creation for picture swap and add to ongoing discussions regarding its appropriate use by comparing these models side by side.


# Deep Fake Generation?
Deep fake generation is the process of producing synthetic media that closely resembles the appearance and behavior of actual people is known as "deep fake generation. This technology can produce incredibly lifelike images, films, and audio. For the intention of limiting the scope of the article, we will focus on pictures swaps. Face swapping is a specific type of deep fake generation that involves replacing one person's face with another person's face in an image or video. 

Face swapping typically entails feeding the network pairs of photos, each of which has the face of one person and the face of another. The network then learns to recognize the features and characteristics of each person's face and the relationship between them.

Once trained, the face swap model can be used to create new pictures or movies with the faces of two persons switched. It is feasible to create highly realistic face swaps that are challenging to discern from actual photographs by carefully tweaking the network's parameters after training it on a sizable collection of facial images.

 ![Overview of Training](/assets/images/team26/Face_swap_ex_left.jpg)
  ![Overview of Training](/assets/images/team26/Face_swap_ex_Right.jpg)
# Deep Face Lab - GAN Model

The piepline has three main components: Extraction, Training, Conversion

 - Extraction is the first phase in DFL, which contains many algorithms and processing parts, i.e. face detection, face alignment, and face segmentation. After the procedure of Extraction, user will get the aligned faces with precise mask and facial landmarks 

 ![Overview of extraction](/assets/images/team26/Overview_extraction.png)

- The training process involves an Encoder and Inter, which have shared weights between the source and destination, along with a separate Decoder for each. This allows for the generalization of the source and destination through the shared Encoder and Inter, effectively solving the unpaired problem. The Inter extracts the latent codes for both the source and destination, denoted as Fsrc and Fdst, respectively.
DFL employs a mixed loss, combining DSSIM (structural dissimilarity) [18] and MSE, to optimize its performance. This combination of losses allows for the benefits of both methods: DSSIM improves generalization for human faces, while MSE enhances clarity. Ultimately, this mixed loss strikes a balance between generalization and clarity.

 ![Overview of Training](/assets/images/team26/Overview_Training.png)

 - The Conversion step allows users to swap faces between the source and destination photos. This is accomplished by transforming the created face and its mask from the destination Decoder to the target image's original place in the source image. The target image is then perfectly blended with the realigned face using a variety of color transfer methods from DFL. The target image and the re-aligned face are combined to create the final blended image.

 ![Overview of Conversion](/assets/images/team26/Overview_Conversion.png)

 # Facewap - VGG16 CNN Model
- The FaceSwap uses the CNN VGG16 model to identify important characteristics of facial features. This process is accompanied by a series of alignment and realignment steps, which are crucial in ensuring the accuracy of the feature extraction. These alignment and realignment steps are an integral part of the overall process of identifying facial features. These faces would then be saved to be used in the next step.
  ![VGG16 Face Dectction](/assets/images/team26/VGG16.png)
- In the train phase, the two outputs are put into an encoder consisting of a NN where they share the same weights and one is decoded to be trained to be as close to the other one as possible. The loss function tries to minimize the error between the two faces and make the swap as natural looking as possible.

- Finally in conversion phase, the CNN is applied again to stick the faces on top of eachother and replace the source face with the target face.

 # Model Comparison Overview 

Deepfakes trained on GANs typically result in results that are more visually convincing than deepfakes trained on conventional CNNs like VGG16. This is due to the fact that GANs were created primarily for creating realistic images by playing an adversarial game between a generator network and a discriminator network. The generator tries to create realistic images while the discriminator tries to differentiate between real and fake images. Over time, the generator learns to produce increasingly realistic images that can fool the discriminator.

Traditional CNNs, such the VGG16, on the other hand, are more commonly employed for classification jobs and might not be designed to produce realistic images. Although they can still be utilized for deepfake generation, they could need more training data or longer training cycles to match GAN-based techniques' levels of visual realism.


# DeepFaceLab Implementation

# FaceSwap Implementation 
The folowing [colab](https://colab.research.google.com/drive/1_j_0N9uCR47ms5paXTKgQVTq1M4GX-9M?usp=sharing) will be used as the setup but there is also a [locally based](https://github.com/deepfakes/faceswap/blob/master/INSTALL.md) installation method based on your hardware configuration.

**NOTE: Colab does not support deepfake training unless you pay premium and keep the session active throughout the whole training prcoess. The local based repo provided by u/deepfake is better or run the repo off a virtual machine with a GPU.**

The process can be divided into three parts:

1. Extract
    
    This step will take photos from an `src` folder and extract faces into an `extract` folder.

2. Train

    This step will take photos from two folders containing pictures of both faces and train a model that will be saved inside the `models` folder.

3. Convert

    This step will take photos from original folder and apply new faces into `modified` folder.



# Quantitative Results Comparison
The following videos below shows results between face swaps between Biden and Trump using both models. Both models seems to perform poorly on Trump's speech while in Biden's speech DeepFakeLab's model is the better model by looking at the nose and lips. DeepFakeLab managed to get Trump's narrower nose and smaller mouth while the FaceSwap model is just jittering over and doesn't really show any Trump-like features.


# Conclusion
It is indeed true that GAN based deepfake models perform better than CNN based models. Although both of the models are not as perfect as the more popular videos, we have limited hardware. We used up our cloud educational credit and had to use a local 3070 with not enough VRAM or time to train the models better. Given more resources it is likely both deepfake models would be better 
# Video Demo
The following are videos of small speeches by Biden and Trump Side-by-Side between their originals and respective model.

## Biden's Speech
### Faceswap
<iframe width="560" height="315" src="https://www.youtube.com/embed/SCYj_pg4sp0" frameborder="0" allowfullscreen></iframe>

### DeepFakeLab
<iframe width="560" height="315" src="https://www.youtube.com/embed/TzlQ1QFRrL4" frameborder="0" allowfullscreen></iframe>

## Trump's Speech
### Faceswap
<iframe width="560" height="315" src="https://www.youtube.com/embed/SteB1xI8-V8" frameborder="0" allowfullscreen></iframe>

### DeepFakeLab
<iframe width="560" height="315" src="https://www.youtube.com/embed/ccITACisf18" frameborder="0" allowfullscreen></iframe>


# References

**First order Motion Model**

- Repository:
 First Order Motion Model. GitHub repository, https://github.com/AliaksandrSiarohin/first-order-model, 2019.

- Paper:
Siarohin, Aliaksandr, Stéphane Lathuilière, Sergey Tulyakov, Elisa Ricci, and Nicu Sebe. (2019). "First Order Motion Model for Image Animation". https://papers.nips.cc/paper/2019/hash/31c0b36aef265d9221af80872ceb62f9-Abstract.html


**Face Swap**
- Repository: DeepFakes. FaceSwap. Github repository,  https://github.com/deepfakes/faceswap, 2018.

- Paper: Korshunova, I., Shi, W., Dambre, J., & Theis, L. (2017). "Fast Face-swap Using Convolutional Neural Networks".  https://arxiv.org/abs/1611.09577.

**DeepFaceLab**
- Repository: DeepFaceLab. GitHub repository, https://github.com/iperov/DeepFaceLab, 2020.
- Paper: Perov, I., Liu, K., Umé, C., Facenheim, C. S., Jiang, J., Zhou, B., Gao, D., Chervoniy, N., Zhang, S., Marangonda, S., dpfks, M., RP, L., Wu, P., & Zhang, W. (2020). "DeepFaceLab: A simple, flexible and extensible face swapping framework". https://arxiv.org/pdf/2005.05535v4.pdf.

**Presidents' Face Datatset**

 - [link](https://www.kaggle.com/datasets/sergiovirahonda/presidentsfacesdataset)


