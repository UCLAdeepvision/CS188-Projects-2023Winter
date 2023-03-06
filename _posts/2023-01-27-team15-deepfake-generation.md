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


## Most popular GANs

GANs have proven to be one of the most effective ways to generate Deep fakes. We will now perform an in depth analysis of three varieties of GANs: Cycle-GAN, Star-GAN, and Style-GAN.

### Cycle-GAN Networks

The Cycle-GAN Networks devised in 2017 promised to perform image to image translation without paired examples. To do so, several assumptions needed to be made:
- There is an underlying relationship between the domains that must be learned
- Translation should be cycle consistent.
- The mapping is stochastic- probabilistic modeling is involved and there is some uncertainty in the output

The model is as follows:
![cycle-gan model]({{ '/assets/images/team-15/3.png' | relative_url }})
{: style="width: 600px; max-width: 100%; padding-top: 5px;"}
*Cycle-GAN model. (Image source: <https://arxiv.org/pdf/1703.10593.pdf>)*


A key component of Cycle-GANs is a DCGAN, which is a convolutional GAN whose generator and discriminator are both CNN’s for training stability purposes. The Cycle-GAN we will study uses two DCGAN’s, which means there are two generators and two discriminators.

Block diagrams of a generator and discriminator in a DGCAN are shown below:
![block diagrams]({{ '/assets/images/team-15/9.png' | relative_url }})
{: style="width: 600px; max-width: 100%; padding-top: 5px;"}
*Block Diagram of a generator. (Image source: <http://noiselab.ucsd.edu/ECE228_2018/Reports/Report16.pdf>)*

![block diagrams]({{ '/assets/images/team-15/10.png' | relative_url }})
{: style="width: 600px; max-width: 100%; padding-top: 5px;"}
*Block Diagram of a discriminator. (Image source: <http://noiselab.ucsd.edu/ECE228_2018/Reports/Report16.pdf>)*

During training, the two DCGANs are sent different sets of images as inputs. We will use the following mathematical terminology:
- The model has two mapping functions: $$G : X \rightarrow Y$$ and $$F : Y \rightarrow X$$
- $$X$$ and $$Y$$ are the domains for each respective DCGAN
- Input images $$x$$ are members of domain $$X$$ and input images $$y$$ are members of domain $$Y$$
- The images generated in domain $$X$$ should be similar to images in domain $$Y$$ and vice versa
- The two generators are $$G$$ and $$F$$ and the respective discriminators are $$D_{y}$$ and $$D_{x}$$
- Note the discriminator for generator $$G$$ is $$D_{y}$$ because we aim to differentiate between fake images $$G(x)$$ and domain $$Y$$
- $$G(x)$$ denotes images generated by $$G$$ and $$F(y)$$ denotes images generated by $$Y$$
&nbsp;  

#### Cycle-GAN Loss

Each DCGAN has two losses. 
- We can combine both adversarial losses together to get:
  - $$\mathcal{L}_{\mathrm{GAN}}=\mathbb{E}_{x \sim p_{\text {data }}(x)}\left[\left(1-D_Y(G(x))\right)^2\right] +\mathbb{E}_{y \sim p_{\text {data }}(y)}\left[\left(1-D_X(F(y))\right)^2\right]$$  
- We can combine both discriminator losses together to get:
  - $$\mathcal{L}_D=\mathbb{E}_{x \sim p_{\text {data }}(x)}\left[D_Y(G(x))^2\right] +\mathbb{E}_{y \sim p_{\text {data }}(y)}\left[D_X(F(y))^2\right]+\mathbb{E}_{x \sim p_{\text {data }}(x)}\left[\left(1-D_X(x)\right)^2\right]  +\mathbb{E}_{y \sim p_{\text {data }}(y)}\left[\left(1-D_Y(y)\right)^2\right]$$  
&nbsp;

**Cycle consistency losses** or $$L_{cyc}$$ are introduced to capture the assumption that the translation should be cycle consistent:
- Forward cycle consistency loss
  - $$x \rightarrow G(x) \rightarrow F(G(x)) \approx x $$  
  - This formula refers to translating an impage $$x$$ from domain $$X$$ to domain $$Y$$ via $$G$$ and then back to domain $$X$$ via $$F$$
- Backward cycle consistency loss  
  - $$y \rightarrow F(y) \rightarrow G(F(y)) \approx y $$  
  - This formula refers to translating an impage $$y$$ from domain $$Y$$ to domain $$X$$ via $$F$$ and then back to domain $$Y$$ via $$G$$
- Total cycle consistency loss  
  - $$ \mathcal{L}_{\text {cyc }}=\mathbb{E}_{x \sim p_{\text {data }}(x)}\left[\|F(G(x))-x\|_1\right] +\mathbb{E}_{y \sim p_{\text {data }}(y)}\left[\|G(F(y))-y\|_1\right] $$  
- The corresponding example code where $$A$$ is $$X$$ and $$B$$ is $$Y$$ is as follows:
```
# GAN loss D_A(G_A(A))
self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
# GAN loss D_B(G_B(B))
self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
# Forward cycle loss || G_B(G_A(A)) - A||
self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
# Backward cycle loss || G_A(G_B(B)) - B||
self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
# combined loss and calculate gradients
self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
self.loss_G.backward()
```
&nbsp;

We can also introduce the **$$L_{identity}$$ loss**. This preserves color composition between inputs and output.  
- $$ \mathcal{L}_{\text {identity }}=\mathbb{E}_{y \sim p_{\text {data }}(y)}\left[\|G(y)-y\|_1\right] +\mathbb{E}_{x \sim p_{\text {data }}(x)}\left[\|F(x)-x\|_1\right] $$  
- The corresponding example code where $$A$$ is $$X$$ and $$B$$ is $$Y$$ is as follows:
```
# Identity loss
    if lambda_idt > 0:
        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        self.idt_A = self.netG_A(self.real_B)
        self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        # G_B should be identity if real_A is fed: ||G_B(A) - A||
        self.idt_B = self.netG_B(self.real_A)
        self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
    else:
        self.loss_idt_A = 0
        self.loss_idt_B = 0
```
&nbsp;

We can create the **total generator loss** as follows:  
$$\mathcal{L}_G=\lambda_1 \mathcal{L}_{\mathrm{GAN}}+\lambda_2 \mathcal{L}_{\text {identity }}+\lambda_3 \mathcal{L}_{\text {cyc }}$$  

Note: $$\lambda_1$$, $$\lambda_2$$, and $$\lambda_3$$ control the relative importances of the various losses.  
&nbsp;

We can also split the **discriminator loss** as well:  
- $$ \mathcal{L}_{D_X}=\mathbb{E}_{x \sim p_{\text {data }}(x)}\left[\left(1-D_X(x)\right)^2\right] +\mathbb{E}_{y \sim p_{\text {data }}(y)}\left[D_X(F(y))^2\right] $$  
- $$ \mathcal{L}_{D_Y}=\mathbb{E}_{y \sim p_{\text {data }}(y)}\left[\left(1-D_Y(y)\right)^2\right] +\mathbb{E}_{x \sim p_{\text {data }}(x)}\left[D_Y(G(x))^2\right] $$  
- The corresponding example code where $$A$$ is $$X$$ and $$B$$ is $$Y$$ is as follows:

```
def backward_D_A(self):
    """Calculate GAN loss for discriminator D_A"""
    fake_B = self.fake_B_pool.query(self.fake_B)
    self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

def backward_D_B(self):
    """Calculate GAN loss for discriminator D_B"""
    fake_A = self.fake_A_pool.query(self.fake_A)
    self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
```
&nbsp;

In each epoch $$ \mathcal{L}_G$$ is calculated and backpropogated. Next, $$ \mathcal{L}_{D_X} $$ is calculated and backpropogated. Lastly, $$\mathcal{L}_{D_Y}$$ is calculated and backpropogated.



## References
[1] Shen, Tianxiang, et al. "“Deep Fakes” using Generative Adversarial Networks (GAN)." *University of California, San Diego*. 2018.

[2] Choi, Yunjey, et al. "StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018.

[3] Ren, Yurui, et al. "Deep Image Spatial Transformation for Person Image Generation." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2020.

---
