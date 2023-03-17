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

Single-Image Super-Resolution (SISR) refers to the process of recovering a high-resolution image from a low-resolution counterpart. This is a classic example of an undetermined inverse problem, as there are multiple possible “solutions” for a given low-resolution image.

![]({{ '/assets/images/team35/intro.png' | relative_url }})
_fig 1. An example of three different image super-resolution models_

We will discuss four methods of SR in this blog post: classical techniques, sparse coding, deep convolutional neural networks, and generative adversarial models.

## Evaluation Metrics

Before diving into the methods of SR, let us first define how we measure the effectiveness of a model.

#### Peak Signal-to-Noise Ratio

Our primary evaluation metric will be Peak Signal-to-Noise Ratio (PSNR). Let us call the ground truth image _I_ and our super-resolved image _K_ where both images are of dimension _m × n_.

First, we calculate the mean squared error between images (MSE):

$$ MSE = \tfrac 1 {m \times n} \sum*{i=0}^{m-1}\sum*{j=0}^{n-1}[I(i,j)-K(i,j)]^2 $$

Then, PSNR is defined, in decibels, as

$$
\begin{align*}
PSNR&=10 \cdot \log_{10} \left(\tfrac{MAX^2_I}{MSE}\right)\\
&= 20 \cdot \log_{10} \left(\tfrac{MAX_I}{\sqrt{MSE}} \right)\\
&= 20 \cdot \log_{10}(MAX_I)-10\cdot \log_{10}(MSE)
\end{align*}
$$

where _MAX_ is the maximum value of a pixel (e.g., for a standard 8-bit image, this would be 255). Note that the formulas above are stated for single-channel images, but are easily generalized to RGB images by simply calculating the average across channels.

Intuitively, a high PSNR implies a better model because it minimizes the MSE between images.

#### Structural Similarity Index Measure

#### Perceptual Similarity

Although PSNR is a theoretically sound evaluation metric, empirical results have shown that it does not effectively capture perceptual similarity. In layman’s terms, although a super-resolved image may have high PSNR, it doesn’t look as realistic to a human eye. This is often because error-based metrics such as PSNR do not account for fine textural detail.

![]({{ '/assets/images/team35/perceptual.png' | relative_url }})
_fig 2. Although the SRResNet model has a higher PSNR, the SRGAN model better preserves textural details in the background._

Since perceptual similarity is a qualitative property, it is measured through mean-opinion-score (MOS). This is essentially the average realism rating of a model’s super-resolved images are across multiple surveys.

## Classical Image Upsampling Techniques

While we will not discuss classical techiniques in detail, we will use them as a baseline to compare the results of our models with. The most common classical techinique for upsampling images is called bicubic interpolation.

![]({{ '/assets/images/team35/bicubic.png' | relative_url }})
_fig 3. Different interpolation methods where the black dot is the interpolated estimate based on the red, yellow, green, and blue samples._

Bicubic interpolation samples 16 in the original low-resolution image for each pixel in the upscaled image. Here’s a real life example where I upsize my profile picture.

![]({{ '/assets/images/team35/bicubicex.png' | relative_url }})
_fig 4. Bicubic interpolation on my profile picture._

Bicubic interpolation and other similar methods are used widely because they are memory efficient and require comparatively less computational power than deep learning models. However, because there is no learning involved, the super-resolved images do not actually recover any information. For instance, if you’ve ever resized an image on a document as I did above, you’ll know that although you can make the image dimension larger, it will make the image blurry.

### Sparse Coding

The sparse coding method is an example-based mathematical method of super-resolution that is an improvement compared to bicubic interpolation because it leverages information provided in training data. Specifically, we discuss Yang et al.’s 2008 sparse coding method.

The essence of this technique is treating the low-resolution image as a downsampled version of its high-resolution counterpart. This was inspired by the idea in signal processing that the original signal can be fully and uniquely recovered from a sampled signal under specific conditions.

Under this assumption, sections of the downsampled image (referred to as patches) can be uniquely mapped to their high-resolution counterpart using a textural dictionary that stores example pairs of low and high resolution image patches.

These dictionaries are generated by randomly sampling patches for training images of similar distribution. For example, if we want to upscale animal images, we generate our training dictionary with images that contain fur, skin, and scale textures. For a given category of images, we need only about 100,000 images which is considerably smaller than previous methods. Once patches are generated, we subtract the mean value of each patch as to represent image texture rather than absolute intensity.

![]({{ '/assets/images/team35/scpatches.png' | relative_url }})
_fig 5. Training images and the patches generated from them._

The algorithm of sparse coding super-resolution is as follows: take overlapping patches of the input image, find the sparse coded approximation of the low-resolution patch, map to a high-resolution patch, combine into a locally consistent recovered image by enforcing that the reconstructed patches agree on areas of overlap.

![]({{ '/assets/images/team35/scalg.png' | relative_url }})
_fig 6. Formal algorithm of sparse coding super-resolution._

The uniqueness property of the mapping comes from the linear algebra of sparse matrices. Essentially, because our textural dictionaries are sufficiently large, we can —any low-resolution image as a linear combination of the training patches. This condition is referred to as having an overcomplete basis. Under certain conditions, the sparsest representation (meaning the linear combination using the least patches) is unique—hence the name sparse coding.

## Deep CNN-based Super Resolution

### SRCNN

At the start of the millennium, artificial intelligence and deep learning became a quickly exploding field of computer science. Because of the shortcomings of mathematical methods, research on image super-resolution began leveraging this newfound power of AI. One of the first pivotal deep learning approaches to image super-resolution was the SRCNN model developed in 2015.

#### Moving Forward from Sparce Coding

The SRCNN is essentially a deep convolutional neural network representation of the aforementioned sparse coding pipeline. Each step of the SC pipeline can be implicitly represented as a hidden layer of the CNN.

##### Patch Extraction and Representation

In sparse coding, this operation extracts overlapping patches from the low-resolution image and represents them as a sparse linear combination of the basis.

With a neural net, this is equivalent to convolving the image with a filter representing the basis. SRCNN convolves the image with multiple filters each of which represents a basis. The selection of these bases is folded into the optimization of the network instead of being hand-selected as it is in sparse coding. This results in a n<sub>1</sub>-dimensional feature for each patch.

The intuition behind having multiple bases stems from wanting a generalizable model. The sparse coding algorithm generates a dictionary based on training examples of similar images (nature images for flowers, furtextures for animals, etc.). In contrast, because a neural net learns the optimal weights, it will implicitly choose the bases that best fit with the training data.

##### Non-linear Mapping

In sparse coding, this operation maps each low-resolution patch to a high-resolution patch.

With a neural net, we want to map each n<sub>1</sub>-dimensional feature to a n<sub>2</sub>-dimensional feature representing the high-resolution patch. This is equivalent to applying n<sub>1</sub> filters of size 1 X 1.

##### Reconstruction

In sparse coding, this operation aggregates all the high-resolution patches to generate the final reconstructed image.

We can think of this as “averaging” the overlapping patches which corresponds to an averaging filter.

![]({{ '/assets/images/team35/cnn.png' | relative_url }})
_fig 7. Sparse coding as a CNN._

Since Yang et. al’s 2008 sparse coding algorithm, many changes have been proposed to improve the speed and accuracy of mapping low-resolution patches to their high-resolution counterparts. However, the dictionary generation and patch aggregation processes are considered pre/post-processing and thus have not been subject to the same optimization. The key advantage of using a CNN for the process is that because all steps of the pipeline are implicitly embedded into the network, they are equally subject to learning as the model finds the optimal weights.

Furthermore, the sparse coding algorithm is an iterative algorithm, taking one pass per patch of the input image. In contrast, SRCNN is a fully feed-forward neural net and it reconstructs the image in a single pass. Below, we can see that SRCNN bypasses the sparse coding algorithm in just a few iterations.

![]({{ '/assets/images/team35/cnnres.png' | relative_url }})
_fig 8. PSNR of SRCNN._

#### Model Architecture

SRCNN generates training samples by first cropping HR images by randomly. A low resolution counterpart is then created by first applying a Gaussian kernel, then applying bicubic downsampling and upsampling to return the LR image to the original size.

SRCNN uses MSE as the loss function between the reconstructed images and the corresponding ground truth images which is minimized using stochastic gradient descent with momentum.

The weights are initialized from a Gaussian distribution with zero mean. The learning rate is 10-4 for the first two layers and 10-5 for the last layer.

The base setting for parameters is _f<sub>1</sub>_ = 9, _f<sub>2</sub>_ = 1, _f<sub>3</sub>_ = 5, _n<sub>1</sub>_ = 64, and _n<sub>2</sub>_ = 32 where fi is the size of the ith filter and ni is as defined above.

During training, to avoid issues with black borders, no padding is used. Thus, the network produces a slightly smaller output. The MSE loss is evaluated only by the pixels that overlap. The network also contains no pooling or fully-connected layer.

Empirical results show that a deeper structure does not necessarily lead to better results, which is not the case for the image classification models we studied in class. However, a larger filter size was found to improve performance.

#### Color Images

Up until now, we have not discussed how to deal with multi-channel images in our SR algorithms. The conventional approach to super-resolve color images is to first transform the images into the YCbCr space. The Y channel is the luminance, while the Cb and Cr channels are the blue-difference and red-difference channels, respectively. Then, we train the proposed algorithm on only the Y channel while the Cb and Cr channels are upscaled through bicubic interpolation.

![]({{ '/assets/images/team35/color.png' | relative_url }})
_fig 9. An image of a mountain separated into its YCbCr channels._

### Our SRCNN Implementation

```
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,  64, kernel_size=9, stride=1, padding=9 // 2),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=5 // 2),
            nn.Conv2d(32, 3,  kernel_size=5, stride=1, padding=5 // 2),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.features(x)
        return x
```

_fig 10. Our PyTorch-based implementation of SRCNN._

#### Deviations From the Paper

Unlike in the paper, we padded each layer such that the output resolution remained the same. This is done as we found that the edge artifacts weren't very noticable after a small number of training epochs.

Additionally, we used the Adam optimizer to optimize the loss with standard values for _β<sub>1</sub>_ and _β<sub>2</sub>_.

For our dataset, instead of training on Set14 or ImageNet as done in the paper, we used a dataset designed for super resolution training from Kaggle.

#### Results

## Generative Adversarial Network-based Super-Resolution

### SRGAN

Although deep convolutional neural networks were a breakthrough in the field of image super-resolution, a key challenge they faced was that they were unable to recover fine textural details. In 2016, a generative adversarial model for super-resolution called SRGAN was proposed that showed significant empirical improvement in perceptual similarity than SRCNN.

![]({{ '/assets/images/team35/ganmanifold.png' | relative_url }})
_fig 11. Patches from the natural image manifold (red) and super-resolved patches obtained with MSE (blue) and GAN (orange). The MSE-based solution appears overly smooth due to the pixel-wise average of possible solutions in the pixel space, while GAN drives the reconstruction towards the natural image manifold producing perceptually more convincing solution._

#### SRGAN vs. SRCNN

Generative adversarial models work by simultaneously training two networks: a generator that is trained to generate the best possible solution and a discriminator which is trained to differentiate between generated solutions and ground truth. The two networks learn jointly in a min-max game as they try to fulfill opposite goals.

Recall our earlier discussion of loss functions, where we noted the difference between perceptual similarity and pixel similarity. SRGAN allows for better recovery of textural detail by defining a loss function that prioritizes perceptual similarity over pixel similarity.

The loss function is a weighted sum of a content loss and an adversarial loss.

The content loss is used to train the generator to generate super-resolved images that can fool the detector. Instead of using a pixel-based MSE loss, we calculate the loss based on the Euclidian distance between feature maps produced by the VGG network of the ground truth image and the super-resolved image.
The adversarial loss is used to train the discriminator network to differentiate between super-resolved images and original photo-realistic images. Specifically, we want to maximize the discriminator network’s perceived probability that the reconstructed image is a natural image. This encourages perceptually superior solutions because we prioritize similarity to natural images instead of only evaluating MSE loss.

The authors of the SRGAN paper speculate that this method of evaluation improves recovery of textural detail because feature maps of deeper layers focus purely on the content while leaving the adversarial loss to manage textural details which are the main difference between CNN-produced super-resolved images and natural images. However, this loss function may not be ideal for all applications. For example, hallucination of finer detail may be less suited for medical or surveillance fields.

#### Model Architecture

The generator network is a residual neural network that uses two convolutional layers with small 3 x 3 kernels and 64 feature maps followed by batch-normalization layers and PReLU as the activation function.

The discriminator network is a feed-forward network that contains eight convolutional layers with an increasing number of 3 x 3 filter kernels, increasing by a factor of 2 from 64 to 512 kernels similar to the VGG network. Strided convolutions are used to reduce the image resolution each time the number of features is doubled. The resulting 512 feature maps are followed by two fully-connected layers and a final sigmoid activation. We use LeakyReLU as the activation (alpha = 0.2) and avoid max-pooling.

![]({{ '/assets/images/team35/srganmodel.png' | relative_url }})
_fig 12. Model architecture for SRGAN discriminator and generator._

### Real-ESRGAN

After SRGAN, an improved version called ESRGAN was released in 2018 with some network tweaks. However, the current state-of-the-art model in image super-resolution is Real-ESRGAN, an improvement upon ESRGAN that was released in 2021. The key difference is that Real-ESRGAN is trained on purely synthetic data that is generated through a multi-step degradation process designed to simulate real-life degradations. In contrast, ESRGAN simply applies a Gaussian filter followed by a bicubic downsampling operation. This method removes artifacts that were previously common when super-resolving common real-world images.

![]({{ '/assets/images/team35/realesrgan.png' | relative_url }})
_fig 13. Comparisons of bicubic-upsampled, ESRGAN, RealSR, and Real-ESRGAN results on real-life images. The Real-ESRGAN model trained with pure synthetic data is capable of enhancing details while removing artifacts for common real-world images._

#### Training image process

Real-ESRGAN uses a high-order degradation model which means that image degradations are modeled with several repeated degradation process. The following are the steps in a single block of the degradation process

- Blur: Apply a Gaussian blur filter (same as SRGAN)
- Noise: Gaussian or Poisson noise filter is added independently to each RGB color channel. Gray noise is also synthesized by adding the same noise to all channels.
- Downsampling: Apply random resize operation chosen from nearest-neighbor interpolation, area resize, bilinear interpolation, and bicubic interpolation.
- JPEG Compression: JPEG compression is a echnique of lossy compression for images. It first converts images into the YCbCr color and then downsamples the Cb and Cr channels. This often results in image artifacts.
- Sinc filters: A sinc filter is used to simulate common ringing and overshoot artifacts in images.

![]({{ '/assets/images/team35/degradation.png' | relative_url }})
_fig 14. Overview of the pure synthetic data generation adopted in Real-ESRGAN. It utilizes a second-order degradation process to model more practical degradations, where each degradation process adopts the classical degradation model._

## Conclusion

![]({{ '/assets/images/team35/timeline.png' | relative_url }})
_fig 15. Timeline of SR methods._

So far in our discussion of the evolution of methods for single image super-resolution, we have covered four key models: sparse coding, SRCNN, SRGAN, and Real-ESRGAN (the current state-of-the-art). Of course, there are other models in the field that we have not been able to cover in detail today. One notable mention is the transformer based model, Efficient Super-Resolution Transformer (ESRT) released last year. Although there has been rapid improvement in the field of super-resolution, there are still many challenges to overcome. Current research is focused on expanding single image super-resolution to video resolution, reducing the amount of data needed to simulate real-world degradation processes, and fine-tuning models for specific SR tasks such as face reconstruction.

## References

[1] Jianchao Yang, J. Wright, T. Huang and Yi Ma, "Image super-resolution as sparse representation of raw image patches," 2008 IEEE Conference on Computer Vision and Pattern Recognition, Anchorage, AK, USA, 2008, pp. 1-8, doi: 10.1109/CVPR.2008.4587647.

[2] Dong, Chao, et al. ["Image Super-Resolution Using Deep Convolutional Networks."](https://paperswithcode.com/paper/image-super-resolution-using-deep) _IEEE Transactions on Pattern Analysis and Machine Intelligence_. 2014.

[3] Ledig, Christian, et al. ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network."](https://paperswithcode.com/paper/photo-realistic-single-image-super-resolution) _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2017.

[4] Wang, X. et al. (2019). ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks. In: Leal-Taixé, L., Roth, S. (eds) Computer Vision – ECCV 2018 Workshops. ECCV 2018. Lecture Notes in Computer Science(), vol 11133. Springer, Cham. https://doi.org/10.1007/978-3-030-11021-5_5

[5] X. Wang, L. Xie, C. Dong and Y. Shan, "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data," 2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), Montreal, BC, Canada, 2021, pp. 1905-1914, doi: 10.1109/ICCVW54120.2021.00217.

[Bicubic Interpolation](https://en.wikipedia.org/wiki/Bicubic_interpolation)

[PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)

[YCbCr](https://en.wikipedia.org/wiki/YCbCr)

[Future of SR](https://towardsdatascience.com/image-super-resolution-an-overview-of-the-current-state-of-research-94294a77ed5a)

---
