---
layout: post
comments: true
title: Inverting Denoising Diffusion Implicit Models
author: Kuan Heng (Jordan) Lin
date: 2022-01-24
---


> We explore the inversion and latent space manipulation of diffusion models, particularly the DDIM, a variation of the DDPM with deterministic sampling and thus a meaningful latent space.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Proposal

Latent spaces are a vital part to modern generative neural networks. Existing methods such as generative adversarial networks (GANs) and variational autoencoders (VAEs) all generate high-dimensional images from some low-dimensional latent space which encodes the features of the generated images. Thus, one can sample---randomly or via interpolation---this latent space to generate unseen images, and with the case of generative adversarial networks (GAN), at relatively high fidelity.

Since the latent space is, in a way, a low-dimensional representation of the generated images, we can think of the generative network as a bijective function $$G : \mathcal{Z} \rightarrow \mathcal{X}$$, where $$\mathcal{Z} \subseteq \mathbb{R}^d$$ is the latent space and $$\mathcal{X} \subseteq \mathbb{R}^n$$ is the image space, $$d \ll n$$. Since the latent space contains the most important visual features of the output images, to manipulate existing images, we can try to invert generative networks to go from the image space to the latent space.

Luckily, this is a very well-researched field. Although finding an analytical solution to $$G^{-1}$$ is difficult, there are many ways to approximate the process, which includes

1. learning-based methods, where we train encoders for networks without one (e.g., GANs),
2. optimization-based methods, where we perform optimization to find the latent vector which best reconstructs the target image, and
3. hybrid methods, where we combine the two methods above, e.g., use learning-based methods to find a good initialization for optimization-based methods [2].

### The Problem With Diffusion Networks

The denoising diffusion probabilistic model (DDPM) is a relatively recent yet incredibly influential advancement in generative neural networks. Particularly, it generates images by iteratively subtracting noise, that is, given some time step $$t$$ (which corresponds to some amount of noise that an image has), the network predicts the noise that was added from $$t - 1$$ to $$t$$. (Consequently, the latent space has the same dimensions as the image space.) The results of DDPMs are on-par with GANs, but it has much greater stability in training as the generator is not trained via adversarial means [3].

Naturally, we ask the question: can we invert DDPMs just like we can with GANs? For the inversion methods to be applied to generative networks, we need two assumptions

1. the latent space represents meaningful image features, and
2. the generator, i.e., $$G$$, is deterministic.

Turns out, DDPMs satisfies neither of these requirements. Since the inference process of DDPMs includes applying noise to the predicted network mean for $$t > 0$$, the generation process is not deterministic. In other words, for some $$\boldsymbol{z} \in \mathcal{Z}$$, $$G(\boldsymbol{z})$$ produces different outputs across different evaluations. Consequently, the latent space does not really have meaningful features.

### Denoising Diffusion Implicit Models

![Human visual cortex system]({{ '/assets/images/team27/non_markovian.png' | relative_url }})
*Figure 1: Illustrated comparison between diffusion (left) and non-Markovian (right) inference models. (The source of the image is [1].)*

The denoising diffusion implicit model (DDIM) is a variation of the DDIM that is "an implicit probabilistic model trained with the DDPM objective." (You see, I would explain this more if I *actually understand the mathematics behind it*---will do that in the near future :P.) Particularly, since the inference/sampling process, i.e., $$G$$, is deterministic for DDIMs, its latent space encodes meaningful image features, demonstrated by its ability to interpolate between images on the latent space level (which is not possible for DDPMs). Also, it allows for much smaller maximum time step $$t$$'s, which drastically speeds up sampling.

Thus, we aim to apply GAN inversion methods to DDIMs and explore if we can either train encoders to reverse the 'diffusion' process, backpropagate gradients through the sampling process, or both. Moreover, with a meaningful latent space, we can further perform dimensionality reduction-based methods on the latent space to control semantic features with methods originally for GANs [4].

## Next Steps

- Actually (try to) understand the mathematics behind DDPMs and, more importantly, DDIMs
- Build a DDIM from scratch (I have already built a DDPM from scratch before) referencing the code repositories below
- Look into GAN inversion methods and *try* to test them

## Code Repositories

[1] [Original PyTorch implementation of DDIM](https://github.com/ermongroup/ddim) (will need some code organization to figure out the important bits)

[2] [Keras implementation of DDIM](https://keras.io/examples/generative/ddim/) (will need to port to PyTorch)

[3] [The annotated diffusion model](https://huggingface.co/blog/annotated-diffusion) (absolutely incredible resource for understanding DDPMs)

[4] [labml.ai annotated deep learning paper implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) (contains DDPM *and* DDIM)


## References

[1] Song, Jiaming, et al. "Denoising Diffusion Implicit Models." *International Conference on Learning Representations*, 2021.

[2] Xia, Weihao, et al. "GAN Inversion: A Survey." *IEEE Transactions on Pattern Analysis & Machine Intelligence*, 2022.

[3] Ho, Jonathan, et al. "Denoising Diffusion Probabilistic Models." *Advances in Neural Information Processing Systems*, 2020, vol. 33, pp. 6840--6851.

[4] Shen, Yujun and Zhou, Bolei. "Closed-Form Factorization of Latent Semantics in GANs." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2021.

---
