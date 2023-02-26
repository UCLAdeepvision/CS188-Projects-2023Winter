---
layout: post
comments: true
title: Deep Fake Generation
author: Sarah Mauricio and Andres Cruz
date: 2023-01-26
---

## Abstract

> The use of deep learning methods in deep fake generation has contributed to the rise of fake of fake images which has some very serious ethical dilemmas. We will look at two different ways to generate deepfake pictures and videos, and will then focus in on Image-to-Image Translation. __ and __ are two different models we will by studying to create deep fake images using Image-to-Image Translation.
<!--more-->
## Table of Contents

* [Introduction](#intro)
* [What is Deepfake](#deepfake)
    * [Example: Image-to-Image Translation](#i2i)
    * [Example: Image Animation](#ia)
* [What is a Generative Adversarial Network (GAN)](#GAN)
* [Cycle GAN](#cycleGAN)
    * [Motivation](#mot1)
    * [Architecture](#arch1)
    * [Architecture Blocks and Code Implementation](#archBlocks1)
    * [Results](#res1)
* [Star GAN](#starGAN)
    * [Motivation](#mot2)
    * [Architecture](#arch2)
    * [Architecture Blocks and Code Implementation](#archBlocks2)
    * [Results](#res2)
* [Demo](#demo)
* [References](#ref)

## Introduction <a name="intro"></a>

We will be working on deep fake generation.

## What is Deepfake <a name="deepfake"></a>

Deepfake is a term used to describe artificially constructed media that portrays an individual or individuals in a way that suits the creator. For example, creating image to image translations, image animations, audio reconstruction, and more. Deepfakes are created using deep neural networks architectures, such as Generative Adversarial Networks or Autoencoders.

### Example: Image-to-Image Translation <a name="i2i"></a>

Image to image translation is the process of extracting features from a source image and emulating those features in another image. An example would be Neural Style Transfer, where a source image is used to create an art style to transfer to another image.

![Style Transfer](/assets/images/team14/style_transfer.png)
* Fig X. Example of Neural Style Transfer (Image source: https://www.v7labs.com/blog/neural-style-transfer)

Below are some common architectures seen for image to image translation.

![image to image](/assets/images/team14/image-to-image.png)
* Fig X. Example of image to image translation architectures (Image source: https://www.researchgate.net/publication/366191121_Enhancing_cancer_differentiation_with_synthetic_MRI_examinations_via_generative_models_a_systematic_review)

### Example: Image Animation <a name="ia"></a>

Image Animation is the action of generating a video where the object from an image is animated using the action from a driving video. For example, if we had an image of a water bottle and a driving video of a ball flying across the screen, the output video would be a water bottle flying across the screen. Thus, it will create an animation based on a single image.

![GAN Flow](/assets/images/team14/pipeline.png)
* Fig X. Example flow of Image Animation

Once applying the model, we would see results similar to the following:

![Image Animation Output](/assets/images/team14/vox-teaser.gif)
* Figure X. Example output from Image Animation


## What is a Generative Adversarial Network (GAN) <a name="GAN"></a>

Generative Adversarial Network, or GAN, is the core frameworkd behind a lot of the DeepFake algorithms you may come across. It is an approach to generate a model for a dataset using deep learning priciples. Generative modeling automatically discovers and learns the patterns in the data so that the model can be used to generate new images that could have been a part of the original dataset. GANs train a generative model that consists of two sub-components: the generator models which is trained to generate new images and the discriminator model which tries to classify an image as real or fake. The generative models and the discriminator model are trained together in an adversarial way, meaning until the discrimnator model classifies images incorrectly about half of the time. This would mean that the generator model generates DeepFake images that could pass as being real.

![GAN Flow](/assets/images/team14/gan1.JPG)
* Fig X. Example of GAN Flow

Below we look into two different models using ideas from GAN.

## Cycle GAN <a name="cycleGAN"></a>

### Motivation <a name="mot1"></a>

![unpaired images](/assets/images/team14/unpaired-images.webp)
* Fig X. Example of paired and unpaired images, (Image source: https://towardsdatascience.com/cyclegan-learning-to-translate-images-without-paired-training-data-5b4e93862c8d)

CycleGAN was used in order to use unpaired image to image translations rather than paired image to image translations. This would allow for more training data and more robust outputs for translations. This model seems to work well on tasks that involve color or texture changes, like day-to-night photo translations, or photo-to-painting tasks like collection style transfer. However, tasks that require substantial geometric changes to the image, such as cat-to-dog translations, usually fail [insert citation].

### Architecture <a name="arch1"></a>


The architecture of CycleGAN consists of a generator taken from Johnson et al [citation here], which consists of 3 convolutional layers, 6 residual block layers, 2 transpose convolutional layers and a final convolution output layer. Should also be noted that all layers similar to Johnson et al are followed by instance normalization.

![CycleGAN Generator](/assets/images/team14/cycleGAN-generator.png)
* Fig X. Example of CycleGAN Generator architecture (Image source: https://towardsdatascience.com/cyclegan-learning-to-translate-images-without-paired-training-data-5b4e93862c8d)

The discriminator uses a 70x70 PatchGAN architecture, which are used to classify 70x70 overlapped images to see if they are real or fake. The PatchGAN architecture consists of 5 convolutional layers with instance normalization [insert citation].

![CycleGAN Discriminator](/assets/images/team14/cycleGAN-discriminator.webp)
* Fig X. Example of CycleGAN Discriminator architecture (Image source: https://towardsdatascience.com/cyclegan-learning-to-translate-images-without-paired-training-data-5b4e93862c8d)
 
 
#### Loss Function

There are two kinds of loss in CycleGANs. There is the regular adversarial loss like other normal GANs:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

Then there is also the cyclic-consistency loss which ensures that F(G(x)) ~ x and G(F(y)) ~ y

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
\mathbf{Loss}_cyc = 
$$
 
### Architecture Blocks and Code Implementation <a name="archBlocks1"></a>

```
# This is a sample code block
```



### Results <a name="res1"></a>

## Star GAN <a name="starGAN"></a>

### Motivation <a name="mot2"></a>
StarGAN is a generative adversarial network that learns the mappings among multiple domains using only a single generator and a discriminator, training effectively from images of all domains (Choi 2). The topology could be represented as a star where multi-domains are connected, thus receiveing the name StarGAN. 

![StarGAN Results](/assets/images/team14/star1.JPG)
* Fig X. Example of multi-domain image-to-image translation on CelebA dataset using StarGAN

StarGAN consists of two modules, a discriminator and a generator. The discriminator learns to differentiate between real and fake images and begins to clssify the real images with its proper domain. The generator takes an image and a target domain label as input and generates a fake image with them. The target domain label is then spatially replicated and concatenated with the image given as input. The generator attempts to reconstruct the orginal image via the fake image when given the original domain label. Lastly, the generator tries to generate images that are almost identical to the real images and will be classified as being from the target domain by the discriminator.

![StarGAN Flow](/assets/images/team14/star2.JPG)
* Fig X. Example flow of StarGAN where D represents the discriminator and G represents the generator

The overarching goal of StarGAN is to translate images from one domain to the other domain. For example, translating an image with a red leaves to an image with yellow leaves.

### Architecture <a name="arch2"></a>

The architecture of StarGAN consists of a generator, which consists of two convolutional layers with a stride size of two for downsampling, six residual blocks, and two tranposed convolutional layers with a stride size of two for upsampling. Instance normalization is also used in all layers except the last. The architecture we use for this is an adaptation of the CycleGAN generator.

![StarGAN Generator](/assets/images/team14/star4.JPG)
* Fig X. Example of StarGAN Generator Architecture (Image source: https://arxiv.org/pdf/1711.09020v3.pdf)

The discriminator uses a . It uses Leaky ReLU with a negative slope of 0.01.

![StarGAN Discriminator](/assets/images/team14/star5.JPG)
* Fig X. Example of StarGAN Discriminator architecture (Image source: https://arxiv.org/pdf/1711.09020v3.pdf)

#### Loss Functions

The discriminator produces probability distributions over source and domain labels as follows:

$$
\mathbf{D} : \mathbf{x}\rightarrow\{{D_{src}(x), D_{cls}(x)}\}
$$

To ensure the generated images are indistinguishable from the real images, adversarial loss is used:

$$
\mathbf{L}_{adv} = \mathbb{E}_{x}[log D_{src}(x)] + \mathbb{E}_{x,c}[log (1-D_{src}(G(x,c)))]
$$

Here, G generates an image G(x,c) that is conditioned on the input image, x, and the target domain label, c. D then tries to determine if it is a real or fake image. 

We need to add an auxiliary classifier on top of our D and impose the domain classification loss when optimizing D and G in order to classify the output image y that was generated from our input image dx.

$$
\mathbf{L}^{r}_{cls} = \mathbb{E}_{x, c'}[-log D_{cls}(c'|x)]
$$

Then, our loss function for domain classification is:

$$
\mathbf{L}^{f}_{cls} = \mathbb{E}_{x, c}[-log D_{cls}(c|G(x,c))]
$$

Now, out adversarial and classification losses are minimized, but this does not guarantee that the translated images preserve the content of its imput images. To reduce this issue, we aneed to apply a cycle consistency loss to G:

$$
\mathbf{L}_{rec} = \mathbb{E}_{x, c, c'}[||x-G(G(x, c), c')||]
$$

Thus, the to optimize G and D we have the following formulas:

$$
\mathbf{L}_{D} = -\mathit{L}_{adv} + {\lambda}_{cls}\mathit{L}^{r}_{cls}
$$

$$
\mathbf{L}_{G} = \mathit{L}_{adv} + {\lambda}_{cls}\mathit{L}^{f}_{cls} + {\lambda}_{rec}\mathit{L}_{rec}
$$

Lastly, we improve our lossfunction to generate higher quality images and to stabilize the trianing process. The new loss formula uses Wasserstein's GAN objective with gradient penalty:

$$
\mathbf{L}_{adv} = \mathbb{E}_{x}[D_{src}(x)] - \mathbb{E}_{x,c}[D_{src}(G(x,c))] - {\lambda}_{gp}\mathbb{E}_{{\hat}{x}}[(||{\delta}_{\hat{x}}\mathit{D}_{src}({\hat}{x})||_{2}-1)^{2}]
$$

### Architecture Blocks and Code Implementation <a name="archBlocks2"></a>

This is the ResidualBlock module.
```
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
```

The code for the Generator is as follows. It uses the ResidualBlock module that is defined above. 

```
class GeneratorResNet(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), res_blocks=9, c_dim=5):
        super(GeneratorResNet, self).__init__()
        channels, img_size, _ = img_shape

        # Initial convolution block
        model = [
            nn.Conv2d(channels + c_dim, 64, 7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        curr_dim = 64
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim *= 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(curr_dim)]

        # Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim = curr_dim // 2

        # Output layer
        model += [nn.Conv2d(curr_dim, channels, 7, stride=1, padding=3), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, c), 1)
        return self.model(x)
```

The code for the Discriminator is as follows.

```
class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), c_dim=5, n_strided=6):
        super(Discriminator, self).__init__()
        channels, img_size, _ = img_shape

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1), nn.LeakyReLU(0.01)]
            return layers

        layers = discriminator_block(channels, 64)
        curr_dim = 64
        for _ in range(n_strided - 1):
            layers.extend(discriminator_block(curr_dim, curr_dim * 2))
            curr_dim *= 2

        self.model = nn.Sequential(*layers)

        # Output 1: PatchGAN
        self.out1 = nn.Conv2d(curr_dim, 1, 3, padding=1, bias=False)
        # Output 2: Class prediction
        kernel_size = img_size // 2 ** n_strided
        self.out2 = nn.Conv2d(curr_dim, c_dim, kernel_size, bias=False)

    def forward(self, img):
        feature_repr = self.model(img)
        out_adv = self.out1(feature_repr)
        out_cls = self.out2(feature_repr)
        return out_adv, out_cls.view(out_cls.size(0), -1)
```



### Results <a name="res2"></a>

## Demo <a name="demo"></a>

## References <a name="ref"></a>

[1] Goodfellow, Ian J., et al. Generative Adversarial Networks. 2014. DOI.org (Datacite), https://doi.org/10.48550/ARXIV.1406.2661.

https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py

[2] Tolosana, Ruben, et al. DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection. 2020. DOI.org (Datacite), https://doi.org/10.48550/ARXIV.2001.00179.

https://github.com/deepfakes/faceswap

[3] Zhang, Tao. “Deepfake Generation and Detection, a Survey.” Multimedia Tools and Applications, vol. 81, no. 5, Feb. 2022, pp. 6259–76. DOI.org (Crossref), https://doi.org/10.1007/s11042-021-11733-y.

[4] Brownlee, Jason. “A Gentle Introduction to Generative Adversarial Networks (Gans).” MachineLearningMastery.com, 19 July 2019, https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/. 

https://github.com/Deepfakes/



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
