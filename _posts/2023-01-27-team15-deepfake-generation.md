---
layout: post
comments: true
title: Deepfake Generation
author: Shrea Chari, Dhakshin Suriakannu
date: 2023-01-24
---


> This blog details Shrea Chari and Dhakshin Suriakannu's project for the course CS 188: Computer Vision. This project discusses the topic of Deepfake Generation.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## What Are "Deep Fakes"?

“Deep Fakes” are synthetic media created using artificial intelligence. In current times, Deep Fake technology has been used to generate “fake” images or videos meant to look like they were captured in the real world. 


![Deepfake Example]({{ '/assets/images/team-15/8.jpeg' | relative_url }})
{: style="width: 600px; max-width: 100%; padding-top: 15px;"}
*Deep Fake made of actor Tom Cruise. (Image source: <https://www.trymaverick.com/blog-posts/are-deep-fakes-all-evil-when-can-they-be-used-for-good>)*


A popular subset of Deep Fakes can be found in face swapped images and videos, as shown above. This blog will detail various approaches to Deep Fake generation and analyze their strengths and weaknesses.

With the ease of access to tools such as AI and the lack of paired training data needed to generate this content, the deceptive prowess of Deep Fakes has become concerning to many around the world. As a result, Deep Fake detection tools and algorithms have become popular as well.

## Generative Adversarial Networks

Generative Adversarial Networks or GAN’s, are the core technology behind most Deep Fake algorithms. GAN’s feature a pair of neural networks- one generator and one discriminator- that are trained simultaneously and “compete” to produce the a realistic Deep Fake. The generator does not have access to training data and only improves through its interaction with the discriminator, whose task is to deduce the probability that a given image is “real.” 

![gan example]({{ '/assets/images/team-15/1.png' | relative_url }})
{: style="width: 600px; max-width: 100%; padding-top: 15px;"}
*Diagram of a generative adversarial network. (Image source: <https://arxiv.org/pdf/1710.07035.pdf>)*

## Image to Image Translation

Image to image translation is a subfield of computer vision problems. The goal of such problems is to learn a mapping between an input and output image using paired training data. Most commonly, image to image translation is done using Convolutional Neural Networks to learn a parametric translation function from a dataset of input and output images. This allows users to generate a new image that has the same essential characteristics as an input image, but is in a different domain or style.

![image to image translation]({{ '/assets/images/team-15/2.jpeg' | relative_url }})
{: style="width: 800px; max-width: 100%; padding-top: 15px;"}
*Image to image translation examples. (Image source: <https://arxiv.org/pdf/1703.10593.pdf>)*

One drawback of such an approach however is that paired training data is not always available. For example, image pairs rarely exists for artistic stylization tasks due to the difficulty of generation: this requires skilled artists to craft artwork by hand. 

## Unpaired Image to Image Translation

An early technique to perform unpaired image to image translation was a Bayesian technique to determine the most likely output image. In this particular approach, P(X) is a patch-based Markov random field taken from the source image. The likelihood of the input is represented by a Bayesian network obtained from multiple style images. 

### CoGAN

In 2016, a Coupled GAN or CoGAN was proposed to learn a joint distribution of multi-domain images. CoGAN features a tuple of GAN’s for each image domain and uses a weight sharing constraint to learn a joint distribution for multi domain images.

**INCLUDE FORMULAS**

As part of the study, CoGAN was used to generate faces with different attributes. The image below shows 

![cogan example]({{ '/assets/images/team-15/4.png' | relative_url }})
{: style="width: 800px; max-width: 100%; padding-top: 15px;"}
*Pair face generation for blond-hair, smiling, and eyeglasses attributes. For each pair, the first row contains the faces with the attributes and the second row contains faces without the attributes. (Image source: <https://arxiv.org/pdf/1606.07536.pdf>)*


Since at the time no models with the same capabilities existed, CoGAN was compared to a conditional GAN, which took a binary variable representing output domain as an input. When applied to specific tasks, CoGAN consistently outperformed the conditional GAN. One one task, CoGAN achieved accuracy of 0.952, while the conditional GAN has accuracy 0.909. On another task, CoGAN had accuracy 0.967 and the conditional GAN 0.778. 

## Neural Style Transfer

Another variation of image to image translation is done with neural style transfer. This technique takes a content image and a style reference and blends the two images together so that the output looks like the content image in the style of the reference. An example can be seen below. 

![dog example]({{ '/assets/images/team-15/5.jpeg' | relative_url }})
{: style="width: 300px; max-width: 100%; padding-top: 15px;"}
![kandinsky example]({{ '/assets/images/team-15/6.jpeg' | relative_url }})
{: style="width: 300px; max-width: 100%; padding-top: 5px;"}
![neural transfer example]({{ '/assets/images/team-15/7.png' | relative_url }})
{: style="width: 300px; max-width: 100%; padding-top: 5px;"}
*Example of neural style transfer featuring a dog and Wassily Kandinsky's Composition 7. (Image source: <https://www.tensorflow.org/tutorials/generative/style_transfer>)*

This can be done with TensorFlow, the steps are as follows (small snippets of code are included for reference):
- Setup
  - Import and Configure Models

    ```
    import os
    import tensorflow as tf`
    # Load compressed models from tensorflow_hub'
    os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED' 
    ```

- Visualize the Input
- Fast Style Transfer using TF-Hub
- Define content and style representations
- Build the model
- Calculate style

```
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)
```

- Extract style and content
- Run gradient descent

```
@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))
```

- Add total variation loss (change train_step)

```
@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)
    loss += total_variation_weight*tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))
```
- Re-run the optimization

## Cycle-GAN Networks

The Cycle-GAN Networks devised in 2017 promised to perform image to image translation without paired examples. To do so, several assumptions needed to be made:
- There is an underlying relationship between the domains that must be learned
- Translation should be cycle consistent: **include math**
- The mapping is stochastic- probabilistic modeling is involved and there is some uncertainty in the output

The model is as follows:
![neural transfer example]({{ '/assets/images/team-15/3.png' | relative_url }})
{: style="width: 600px; max-width: 100%; padding-top: 5px;"}
*Cycle-GAN model. (Image source: <https://arxiv.org/pdf/1703.10593.pdf>)*


The model has two mapping functions- $$G : X \rightarrow Y$$ and $$F : Y \rightarrow X$$- and adversarial discriminators $$Dx$$ and $$Dy$$. $$Dx$$ encourages $$F$$ to translate $$Y$$ into outputs indistinguishable from domain $$X$$ and vice versa for $$Dy$$ and $$G$$.

Cycle consistency losses are introduced to capture the assumption that the translation should be cycle consistent:
- Forward cycle consistency loss
  - $$x \rightarrow G(x) \rightarrow F(G(x)) \approx x $$
- Backward cycle consistency loss
  - $$y \rightarrow F(y) \rightarrow G(F(y)) \approx y $$


## References
[1] Shen, Tianxiang, et al. "“Deep Fakes” using Generative Adversarial Networks (GAN)." *University of California, San Diego*. 2018.

[2] Choi, Yunjey, et al. "StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018.

[3] Ren, Yurui, et al. "Deep Image Spatial Transformation for Person Image Generation." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2020.

---
