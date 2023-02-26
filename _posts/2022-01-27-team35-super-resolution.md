---
layout: post
comments: true
title: Evolution of Image Super-Resolution Techniques
author: Ethan Truong, Archisha Datta
date: 2022-01-28
---

> Image super-resolution is a process used to upscale low-resolution images to higher resolution images while preserving texture and semantic data. We will outline how state-of-the art techniques have evolved over the last decade and compare each model to its predecessor. We will also implement the CNN-based approach to super-resolution and test it on our own images.

<!--more-->

{: class="table-of-content"}

## Table of Contents

{:Table of Contents}

- Introduction
  - Evaluation Metrics
    - PSNR
    - Perceptual Similarity
- Mathematical Methods
- Sparse Coding
- Deep Convolutional Neural Networks (SRCNN)
- Generative Adversarial Networks (SRGAN)
  - SRGAN vs. SRCNN
  - Model Architecture
- Generative Adversarial Networks 2.0 (ESRGAN)
  - SRGAN vs. ESRGAN
  - Training Process
- Our Implementation
  - Model Architecture
  - Results
- Conclusion
- Works Cited

---

### **Current code/algorithms can be found [here](https://colab.research.google.com/drive/1LGX6pKKAc63K1OlgVmN7U_unzG78LAKw?usp=sharing).**

## Introduction

Image super-resolution (SR) refers to the process of recovering a high-resolution image from its low-resolution counterpart. This is a classic example of an undetermined inverse problem, as there are multiple possible “solutions” for a given low-resolution image.

![]({{ '/assets/images/team35/intro.png' | relative_url }})
_f<sub>i</sub>g 1. An example of three different image super-resolution models_

We will discuss four methods of SR in this blog post: mathematical methods, sparse coding, deep convolutional neural networks, and generative adversarial models.

### Evaluation Metrics

Before diving into the methods of SR, let us f<sub>i</sub>rst def<sub>i</sub>ne how we measure the effectiveness of a model.

#### PSNR

Our primary evaluation metric will be PSNR, or Peak Signal-to-Noise Ratio. Let us call the ground truth image I and our super-resolved image K where both images are of dimension m x n.

f<sub>i</sub>rst, we calculate the mean squared error between images (MSE):

$$ MSE = \tfrac 1 {mn} \sum \_{i=0}^{m-1}\sum \_{j=0}^{n-1}[I(i,j)-K(i,j)]^2 $$

Then, PSNR is def<sub>i</sub>ned, in decibels, as

$$
\begin{align*}
PSNR&=10 \cdot \log_{10} \left(\tfrac{MAX^2_I}{MSE}\right)\\
&= 20 \cdot \log_{10} \left(\tfrac{MAX_I}{\sqrt{MSE}} \right)\\
&= 20 \cdot \log_{10}(MAX_I)-10\cdot \log_{10}(MSE)
\end{align*}
$$

where MAX is the maximum value of a pixel. For a standard 8-bit image, this would be 255. Note that the formulas above are stated for single-channel images, but are easily generalized to RGB images by simply calculating the average across channels.

Intuitively, a high PSNR implies a better model because it minimizes the MSE between images.

#### Perceptual Similarity

Although PSNR is a theoretically sound evaluation metric, empirical results have shown that it does not effectively capture perceptual similarity. In layman’s terms, although a super-resolved image may have high PSNR, it doesn’t look as realistic to a human eye. This is often because error-based metrics such as PSNR do not account for f<sub>i</sub>ne textural detail.

![]({{ '/assets/images/team35/perceptual.png' | relative_url }})
_f<sub>i</sub>g 2. Although the SRResNet model has a higher PSNR, the SRGAN model better preserves textural details in the background._

Since perceptual similarity is a qualitative property, it is measured through mean-opinion-score (MOS). This is essentially the average realism rating of a model’s super-resolved images are across multiple surveys.

### Mathematical Methods

Although we will not discuss mathematical models of super-resolution in detail, it is important to acknowledge their existence since they are the base method that we will compare future models to.

The most common mathematical interpolation method is bicubic interpolation. This is a 2D extension of cubic interpolation.

![]({{ '/assets/images/team35/bicubic.png' | relative_url }})
_f<sub>i</sub>g 3. Different interpolation methods where the black dot is the interpolated estimate based on the red, yellow, green, and blue samples._

In our case, bicubic interpolation each additional pixel in the high-resolution image using the original pixels in the low-resolution image as sample points. Here’s a real life example where I upsize my prof<sub>i</sub>le picture.

![]({{ '/assets/images/team35/bicubicex.png' | relative_url }})
_f<sub>i</sub>g 4. Bicubic interpolation on my prof<sub>i</sub>le picture._

Bicubic interpolation and other similar methods are used widely because they are memory eff<sub>i</sub>cient and require comparatively less computational power than deep learning models. However, because there is no learning involved, the super-resolved images do not actually recover any information. For instance, if you’ve ever resized an image on a document as I did above, you’ll know that although you can make the image dimension larger, it will make the image blurry.

### Sparse Coding

The sparse coding method is an example-based mathematical method of super-resolution that is an improvement compared to bicubic interpolation because it leverages information provided in training data. Specif<sub>i</sub>cally, we discuss Yang et al.’s 2008 sparse coding method.

The essence of this technique is treating the low-resolution image as a downsampled version of its high-resolution counterpart. This was inspired by the idea in signal processing that the original signal can be fully and uniquely recovered from a sampled signal under specif<sub>i</sub>c conditions.

Under this assumption, sections of the downsampled image (referred to as patches) can be uniquely mapped to their high-resolution counterpart using a textural dictionary that stores example pairs of low and high resolution image patches.

These dictionaries are generated by randomly sampling patches for training images of similar distribution. For example, if we want to upscale animal images, we generate our training dictionary with images that contain fur, skin, and scale textures. For a given category of images, we need only about 100,000 images which is considerably smaller than previous methods. Once patches are generated, we subtract the mean value of each patch as to represent image texture rather than absolute intensity.

![]({{ '/assets/images/team35/scpatches.png' | relative_url }})
_f<sub>i</sub>g 5. Training images and the patches generated from them._

The algorithm of sparse coding super-resolution is as follows: take overlapping patches of the input image, f<sub>i</sub>nd the sparse coded approximation of the low-resolution patch, map to a high-resolution patch, combine into a locally consistent recovered image by enforcing that the reconstructed patches agree on areas of overlap.

![]({{ '/assets/images/team35/scalg.png' | relative_url }})
_f<sub>i</sub>g 6. Formal algorithm of sparse coding super-resolution._

The uniqueness property of the mapping comes from the linear algebra of sparse matrices. Essentially, because our textural dictionaries are suff<sub>i</sub>ciently large, we can —any low-resolution image as a linear combination of the training patches. This condition is referred to as having an overcomplete basis. Under certain conditions, the sparsest representation (mean—ing the linear combination using the least patches) is unique—hence the name sparse coding.

### Deep CNNs (SRCNN)

At the start of the millennium, artif<sub>i</sub>cial intelligence and deep learning became a quickly exploding f<sub>i</sub>eld of computer science. Because of the shortcomings of mathematical methods, research on image super-resolution began leveraging this newfound power of AI. One of the f<sub>i</sub>rst pivotal deep learning approaches to image super-resolution was the SRCNN model developed in 2015.

#### Sparse Coding vs. CNNs

The SRCNN is essentially a deep convolutional neural network representation of the aforementioned sparse coding pipeline. Each step of the SC pipeline can be implicitly represented as a hidden layer of the CNN.

##### Patch Extraction and Representation

In sparse coding, this operation extracts overlapping patches from the low-resolution image and represents them as a sparse linear combination of the basis.

With a neural net, this is equivalent to convolving the image with a f<sub>i</sub>lter representing the basis. SRCNN convolves the image with multiple f<sub>i</sub>lters each of which represents a basis. The selection of these bases is folded into the optimization of the network instead of being hand-selected as it is in sparse coding. This results in a n<sub>1</sub>-dimensional feature for each patch.

The intuition behind having multiple bases stems from wanting a generalizable model. The sparse coding algorithm generates a dictionary based on training examples of similar images (nature images for flowers, furtextures for animals, etc.). In contrast, because a neural net learns the optimal weights, it will implicitly choose the bases that best f<sub>i</sub>t with the training data.

##### Non-linear Mapping

In sparse coding, this operation maps each low-resolution patch to a high-resolution patch.

With a neural net, we want to map each n<sub>1</sub>-dimensional feature to a n<sub>2</sub>-dimensional feature representing the high-resolution patch. This is equivalent to applying n<sub>1</sub> f<sub>i</sub>lters of size 1 X 1.

##### Reconstruction

In sparse coding, this operation aggregates all the high-resolution patches to generate the f<sub>i</sub>nal reconstructed image.

We can think of this as “averaging” the overlapping patches which corresponds to an averaging f<sub>i</sub>lter.

![]({{ '/assets/images/team35/cnn.png' | relative_url }})
_f<sub>i</sub>g 7. Sparse coding as a CNN._

Since Yang et. al’s 2008 sparse coding algorithm, many changes have been proposed to improve the speed and accuracy of mapping low-resolution patches to their high-resolution counterparts. However, the dictionary generation and patch aggregation processes are considered pre/post-processing and thus have not been subject to the same optimization. The key advantage of using a CNN for the process is that because all steps of the pipeline are implicitly embedded into the network, they are equally subject to learning as the model f<sub>i</sub>nds the optimal weights.

Furthermore, the sparse coding algorithm is an iterative algorithm, taking one pass per patch of the input image. In contrast, SRCNN is a fully feed-forward neural net and it reconstructs the image in a single pass. Below, we can see that SRCNN bypasses the sparse coding algorithm in just a few iterations.

![]({{ '/assets/images/team35/cnnres.png' | relative_url }})
_f<sub>i</sub>g 8. PSNR of SRCNN._

#### Model Architecture

We preprocess the images by upscaling to the desired size using bicubic interpolation.

We def<sub>i</sub>ne the loss function to minimize as the MSE between the reconstructed images and the corresponding ground truth images. The loss is minimized using stochastic gradient descent with momentum and standard backpropogation.

The weights are initialized from a Gaussian distribution with zero mean. The learning rate is 10-4 for the f<sub>i</sub>rst two layers and 10-5 for the last layer.

The base setting for parameters is f<sub>1</sub> = 9, f<sub>2</sub> = 1, f<sub>3</sub>= 5, n<sub>1</sub> = 64, and n<sub>2</sub> = 32 where f<sub>i</sub> is the size of the ith f<sub>i</sub>lter and n<sub>i</sub> is as def<sub>i</sub>ned above.

During training, to avoid issues with black borders, no padding is used. Thus, the network produces a slightly smaller output. The MSE loss is evaluated only by the pixels that overlap. The network also contains no pooling or fully-connected layer.

Empirical results show that a deeper structure does not necessarily lead to better results, which is not the case for the image classif<sub>i</sub>cation models we studied in class. However, a larger f<sub>i</sub>lter size was found to improve performance.

#### Color Images

Up until now, we have not discussed how to deal with multi-channel images in our SR algorithms. The conventional approach to super-resolve color images is to f<sub>i</sub>rst transform the images into the YCbCr space. The Y channel is the luminance, while the Cb and Cr channels are the blue-difference and red-difference channels, respectively. Then, we train the proposed algorithm on only the Y channel while the Cb and Cr channels are upscaled through bicubic interpolation.

![]({{ '/assets/images/team35/color.png' | relative_url }})
_f<sub>i</sub>g 9. An image of a mountain separated into its YCbCr channels.._

### Generative Adversarial Networks (SRGAN)

SRGAN vs. SRCNN
Recover f<sub>i</sub>ner textural details by prioritizing perceptual similarity over pixel-similarity
ex) higher PSNR but not better image
Model Architecture
Training
Perceptual Loss
Adversarial loss
Content loss
Calculated on feature maps of VGG network
Discriminator network trained to distinguish upscaled images from real high-res images
Architecture
See paper

ESRGAN
SRGAN vs. ESRGAN
Trained on synthetic data
Training image process
Higher order degradation: repeat degradation process
Blur
Noise
Resize
JPEG Compression

Our Implementation
Introduction
What we did
Model Architecture
Diagram
Code snippets
Results
Results on our own images
PSNR
Conclusion
Future in super-resolution research

## Relevant Research Papers

1. #### Learning a Single Convolutional Super-Resolution Network for Multiple Degradations

   ([Code](https://github.com/cszn/SRMD)) Uses CNNs to upscale images with non-standard degradation. This method is more scalable to data encountered in the real world. [1]

2. #### Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

   ([Code](https://github.com/tensorlayer/srgan)) Addresses the issue of retaining f<sub>i</sub>ne textural details during the upscaling process by using a loss function motivated by perceptual similarity rather than pixel similarity. [2]

3. #### Image Super-Resolution Using Deep Convolutional Networks
   ([Code](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)) Deep neural network approach to image super-resolution. [3]

## References

[1] Zhang, Kai, et al. ["Learning a Single Convolutional Super-Resolution Network for Multiple Degradations."](https://paperswithcode.com/paper/learning-a-single-convolutional-super) _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2018.

[2] Ledig, Christian, et al. ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network."](https://paperswithcode.com/paper/photo-realistic-single-image-super-resolution) _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2017.

[3] Dong, Chao, et al. ["Image Super-Resolution Using Deep Convolutional Networks."](https://paperswithcode.com/paper/image-super-resolution-using-deep) _IEEE Transactions on Pattern Analysis and Machine Intelligence_. 2014.

---
