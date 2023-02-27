---
layout: post
comments: true
title: Semantic Image Editing
author: Kevin Huang, Brian Compton
date: 2022-01-29
---


# Introduction
Image generation has the potential to greatly streamline creative processes for professional artists. While current capabilities of image generators have shown to be impressive, these models aren't able to produce good results without fail. However, slightly tweaking text prompts can often result in wildly different images. Semantic image editing has the potential to alleviate this problem, by letting users of image generation models to slightly tweak results in order to achieve higher quality end results. This project will explore and compare some technologies currently available for semantic image editing.

# Semantic Guidance (SEGA)
One option for semantic image editing is semantic guidance. This method bases its functionality on the idea that operations performed in semantic space can result in predictable semantic modifications in the final generated image. A well known example of this can be seen in language prediction: under the word2vec text embedding, the vector representation of 'King - male + female' results in 'Queen'. 

## Guided Diffusion
SEGA is built upon diffusion for image generation. In diffusion for text-to-image generation, the model is conditioned on a text prompt, and iteratively denoises a Gaussian distribution towards an image that accurately reflects the prompt. Under guided diffusion, this process is influenced in order to "guide" the diffusion in specific directions. This method comes with several advantages over other semantic editors, in that it requires no additional training, no architecture extensions, and no external guidance. In particular, the training objective of a diffusion model $\hat{x}$ is:
$$\mathbb{E}_{x,c_p,\epsilon,t}[\omega_t || \hat{x}_\theta(\alpha_t x + \omega_t \epsilon, c_p) - x ||^2_2]$$

Where $(x, c_p)$ is conditioned on text prompt $p$, $t$ is sampled from uniform distribution $t \sim \mathcal{U}([0,1])$, $\epsilon$ is sampled from Gaussian distribution $\epsilon \sim \mathcal{N}(0,I)$, and $\omega_t,\alpha_t$ influence image fidelity based on $t$. The model is trained to denoise $z_t := x + \epsilon$, yielding $x$ with squared error loss. 


Additionally, classifier-free guidance is used for conditioning, resulting in score estimates for the predicted $x$ such that:
$$\tilde{\epsilon}_\theta := \epsilon_\theta(z_t) + s_g(\epsilon_\theta(z_t, c_p) - \epsilon_\theta(z_t))$$

where $s_g$ scales the extent of adjustment, and $\epsilon_\theta$ defines the noise estimates with parameters $\theta$. 

During inference, the model is sampled using $x = (z_t - \tilde{\epsilon}_\theta)$

At a high level, guided diffusion can be explained from the following figure, illustrating the $\epsilon$-space ($\epsilon \sim \mathcal{N}(0,I)$) during diffusion, and the semantic spaces for the concepts involved in editing a prompt "a portrait of a king".

![High to low ]({{ '/assets/images/team16/1.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 1. Semantic space under semantic guidance applied to an image with description "a portrait of a king"* [2].

Here, the unconditioned noise estimate is represented by a black dot, which starts at a random point without semantic grounding. On executing the prompt "a portrait of a king", the black dot is guided (blue vector) to the space intersecting the concepts 'royal' and 'male', resulting in an image of a king. Under guided diffusion, vectors representing the concepts 'male' and 'female' (orange/green vectors) can be made using estimates conditioned on the respective prompts. After subtracting 'male' and adding 'female' to the guidance, the black dot arrives at a point in the space intersecting the concepts 'royal' and 'male', resulting in an image of a queen. 

![High to low ]({{ '/assets/images/team16/2.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 2. Result of guidance operation using 'king' - 'male' + 'female'* [2].

## Semantic isolation
A noise estimate $\epsilon_\theta(z_t, c_e$ is calculated, conditioned on description $e$. We calculate the difference between this, and the unconditioned estimate $\epsilon_\theta(z_t)$, and scale the difference. The numerical values of the resulting latent vector are Gaussian distributed, and the dimensions in the upper and lower tail of the distribution encode the target concept. 1-5% of these dimensions is sufficient to correctly modify the image. Additionally, these concept vectors can be applied simultaneously. 

### Guidance in a single direction (single concept)
For understanding, we can examine guidance in a single direction, in order to build understanding for guidance in multiple directions.
Here, three $\epsilon$-predictions are used to move the unconditioned score estimate $\epsilon_\theta(z_t)$ towards the prompt conditioned estimate $\epsilon_\theta(z_t, c_p)$ and away/towards the concept conditioned estimate $\epsilon_\theta(z_t, c_p)$:
$$\epsilon_\theta(z_t) + s_g(\epsilon_\theta(z_t, c_p) - \epsilon_\theta(z_t)) + \gamma(z_t, c_e)$$
Here, the semantic guidance term ($\gamma$) is defined as $$\gamma(z_t, c_e) = \mu(\psi;s_e, \lambda)\psi(z_t,c_e)$$
where $\psi$ is defined by the editing direction:
$$\psi(z_t, c_p, c_e) = 
    \begin{cases}
        \epsilon_\theta(z_t, c_e) - \epsilon_\theta(z_t) & \text{if positive guidance} \\
        -(\epsilon_\theta(z_t, c_e) - \epsilon_\theta(z_t)) & \text{if negative guidance}
    \end{cases}$$

and $\mu$ applies the editing guidance scale $s_e$ element-wise, and setting values outside of a percentile threshold $\lambda$ to 0:
$$\mu(\psi;s_e, \lambda) = 
\begin{cases}
  s_e & \text{where} |\psi| \geq \eta_\lambda (|\psi|)\\
  0 & \text{otherwise} 
\end{cases}$$
(Here, $\eta_\lambda(\psi)$ is the $\lambda$-th percentile of $\psi$)

Additionally, SEGA uses two additional adjustments:
 - a warm-up parameter $\delta$ to only apply guidance after a warm-up period ($\gamma(z_t, c_p, c_s) := 0$ if $t < \delta$)
- a momentum term $\nu$ to accelerate guidance after iterations where guidance occurred in the same direction. 
Finally, we have:
$$\gamma(z_t, c_e) = \mu(\psi;s_e, \lambda) \psi(z_t, c_e) + s_m \nu_t$$
with momentum scale $s_m \in [0,1]$ and $\nu$ updated as $\nu_t+1 = \beta_m \nu_t + (1 - \beta_m) \gamma_t$

### Guidance in multiple directions (multiple concepts)
For each concept $e_i$, $\gamma^i_t$ is calculated with its own hyperparameter values $\lambda^i, s^i_e$. To incorporate multiple concepts, we will take the weighted sum of all $\gamma^i_t$:
$$
\hat{\gamma_t}(z_t,c_p;e) = \sum_{i\in I}g_i \gamma^i_t(z_t, c_p, c_{e_i})
$$
Each $\gamma^i_t$ may have its own warmup period. Henc, $g_i$ is defined as $g_i = 0$ if $t < \delta_i$. Unlike warmup period, momentum is calculated using all concepts, and applied once all warm-up periods are completed. 

## Demonstration

We have included a colab notebook demonstrating SEGA (more demos to come) \
[Demo Colab](https://colab.research.google.com/drive/1SPfwN8EOAUBfbxqe6Qw4ckNVgPRueCwU?usp=sharing)






<!--more-->

## Three Relevant Research Papers
1. ##### EditGAN: High-Precision Semantic Image Editing
  - [Paper] https://arxiv.org/abs/2111.03186
  - [Code] https://github.com/nv-tlabs/editGAN_release
2. ##### The Stable Artist: Steering Semantics in Diffusion Latent Space
  - [Paper] https://arxiv.org/abs/2301.12247
  - [Code] https://github.com/ml-research/semantic-image-editing
3. ##### Context-Consistent Semantic Image Editing with Style-Preserved Modulation 
  - [Paper] https://arxiv.org/abs/2207.06252
  - [Code] https://github.com/WuyangLuo/SPMPGAN



## Reference
[1] Ling, Huan, et al. EditGAN: High-Precision Semantic Image Editing. 1, arXiv, 2021, doi:10.48550/ARXIV.2111.03186.

[2] Brack, Manuel, et al. SEGA: Instructing Diffusion using Semantic Dimensions. 1, arXiv, 2022, doi:10.48550/arXiv.2301.12247.

[3] Luo, Wuyang, et al. Context-Consistent Semantic Image Editing with Style-Preserved Modulation. 1, arXiv, 2022, doi:10.48550/ARXIV.2207.06252.

---
