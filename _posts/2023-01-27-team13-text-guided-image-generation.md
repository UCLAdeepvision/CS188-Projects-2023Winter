---
layout: post
comments: true
title: Text Guided Image Generation
author: Isaac Li, Ivana Chang
date: 2022-01-27
---


> Diffusion models have been shown to be extremely powerful for image synthesis. In this article, we take an in depth look at diffusion models as a means of generating realistic and relevant images from natural language through two models from OpenAI: GLIDE and DALL-E 2.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Diffusion Models
Diffusion models have been shown to be extremely effective in image synthesis, even outperforming current state-of-the-art generative models like GANs ([Dhariwal and Nichol 2021](https://papers.nips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf)). GANs previously held state of the art on most image generation tasks, yet struggled with scalability and applications to new domains. Diffusion models are a class of likelihood-based models that can produce these high quality images while overcoming difficulties in other models like scalability and distribution coverage or slow synthesis speeds.

As a high level overview, diffusion models involve two main steps: a forward diffusion process and reverse diffusion process. The key idea is to gradually add noise to input samples through a Markov chain of diffusion steps and to train a model to learn how to reverse the diffusion process in order to construct realistic images from noise.

Here is a helpful graphical model depicting the process:
![Diffusion process]({{ '/assets/images/team13/diffusion_process.png' | relative_url }})
*Fig 1: Diffusion Process [3].*

### Forward Diffusion Process
In the forward pass, given a sample selected from the data distribution, $$\mathbf{x}_0 \sim q(\mathbf{x}_0)$$, small amounts of noise are added to the sample in $$T$$ steps, according to a variance schedule $$\beta_1, \dots, \beta_T$$, producing a sequence of noisy samples $$\mathbf{x}_1, \dots, \mathbf{x}_T$$:

$$ \begin{align}q(\mathbf{x}_{1:T}|\mathbf{x}_0):=\prod_{t=1}^T q(\mathbf{x}_t|\mathbf{x}_{t-1})\end{align}$$

$$q(\mathbf{x}_t|\mathbf{x}_{t-1}) := \mathcal{N}(\mathbf{x}_t;\sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

After using the notation $$\alpha_t := 1-\beta_t$$ and $$\bar{\alpha}_t:=\prod_{s=1}^t \alpha_s$$ and reparameterizing, we can sample $$\mathbf{x}_t$$ at any time step:

$$q(\mathbf{x}_t|\mathbf{x}_0)= \mathcal{N}(\mathbf{x}_t;\sqrt{\bar\alpha_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

### Reverse Diffusion Process
In the reverse diffusion process, the diffusion model seeks to reverse the forward process, sampling from $$q(\mathbf{x}_{t-1}|\mathbf{x}_t)$$ in order to recreate the true sample from Gaussian noise input. The reverse process is also defined by a Markov chain, with learned Gaussian transitions beginning with $$p(\mathbf{x}_T)=\mathcal{N}(\mathbf{x}_T;0,\mathcal{I})$$:

$$p_\theta(\mathbf{x}_{0:T}) := p(\mathbf{x}_T)\prod_{t=1}^T p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$$

$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) := \mathcal{N}(\mathbf{x}_{t-1};\mu_\theta(\mathbf{x}_t, t), \sum_\theta(\mathbf{x}_t, t))$$

Note that if the magnitude of the noise added at each step, $$1-\alpha_t$$, is small enough, the posterior can be approximated by a diagonal Gaussian. Additionally, if the total noise added during the forward process is large enough, $$\mathbf{x}_T$$ can be approximated by $$\mathcal{N}(0, \mathcal{I})$$. With this, we can approximate the true posterior:

$$ p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) := \mathcal{N}(\mu_\theta(\mathbf{x}_t), \sum_\theta(\mathbf{x}_t))$$

### Training
Training a diffusion model can be done by optimizing the variational lower bound (VLB) on negative log likelihood [[3]](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf). GLIDE optimizes a surrogate objective function which reweights terms in the VLB. To compute this function, GLIDE generates samples $$\mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_0)$$ by adding Gaussian noise $$\epsilon$$ to $$\mathbf{x}_0$$, then training a model $$\epsilon_\theta$$ to predict the added noise using standard mean-squared error loss:

$$L_\textrm{simple} := E_{t\sim[1,T], x_0\sim q(x_0), \epsilon\sim \mathcal{N}(0, \mathbf{I})} ||\epsilon - \epsilon_\theta(\mathbf{x}_t, t)||^2$$

[Here](https://iterative-refinement.github.io/assets/cascade_movie2_mp4.mp4) is a helpful visual from Google on image generation using diffusion models.

## GLIDE
GLIDE pairs diffusion models with a guidance strategy to improve the quality of samples at the cost of diversity. The two strategies explored are classifier-free and classifier based guidance.

### Classifier-free Guidance
[Ho & Salimans (2021)](https://openreview.net/pdf?id=qw8AKxfYbI) proposed classifier-free diffusion guidance which removes the need for training a separate classifier model, simplifying the training pipeline. GLIDE implements classifier-free guidance by replacing labels in a class-conditional diffusion model $$\epsilon_\theta(x_t|y)$$ with the null label $$\emptyset$$ with a fixed probability during training. When sampling, outputs of the model are guided towards the true label $$y$$:

$$\hat \epsilon_\theta(x_t|y) = \epsilon_\theta(x_t|\emptyset) + s \cdot (\epsilon_\theta(x_t|y) - \epsilon_\theta(x_t|\emptyset))$$

A similar method follows for classifier-free guidance with generic text-based prompts; instead of labels $$y$$, we have captions $$c$$.

### Classifier Guidance
GLIDE makes use of the CLIP model ([Radford et al. 2021](http://proceedings.mlr.press/v139/radford21a/radford21a.pdf)) in classifier guided diffusion. CLIP consists of two pieces: an image encoder $$f(x)$$ and a caption encoder $$g(c)$$. It trains a model that rewards pairs $$(x, c)$$ where the image $$x$$ closely matches the given caption $$c$$. GLIDE applies a CLIP model trained on noisy images to a diffusion model by perturbing the mean $$\mu_\theta(x_t|c)$$ and variance $$\sum_\theta(x_t|c)$$ with the gradient of the dot product of the image and caption encodings with respect to the image:

$$\hat\mu_\theta(x_t|c)=\mu_\theta(x_t|c)+s \cdot \sum_\theta(x_t|c)\nabla_{x_t} (f(x_t)\cdot g(c))$$

### Model Architecture
GLIDE adopts the Ablated Diffusion Model architecture and augments it with text conditioning information. For each noised image $$x_t$$ and corresponding caption $$c$$, the model predicts $$p(x_{t-1}|x_t, c)$$. Captions are encoded into a sequence of $$K$$ tokens and fed into a Transformer model, whose output is used in two ways. The final token embedding is used in place of a class embedding in the ADM model, and the last layer of token embeddings is concatenated to the attention context at each layer in the ADM model [[1](https://arxiv.org/pdf/2112.10741.pdf)].

It then uses the ImageNet 64 x 64 model architecture from Dhariwal & Nichol (2021), scaled to 512 channels resulting in 2.3 billion parameters for the visual part of the model. For the text encoding Transformer, 24 residual blocks of width 2048 are used, resulting in about 1.2 billlion parameters [[1](https://arxiv.org/pdf/2112.10741.pdf)].

Finally, a 1.5 billion parameter upsampling difusion model is trained to upsample images from 64 x 64 to 256 x 256 resolution [[1](https://arxiv.org/pdf/2112.10741.pdf)].

### Results and Limitations
Visually, the results of GLIDE are quite impressive. Classifier-free guidance appears to generate more realistic images than those produced using CLIP guidance. Sample results of GLIDE with classifier-free guidance are shown in Figure 2.
![GLIDE samples]({{ '/assets/images/team13/GLIDE-samples.png' | relative_url }})
{: style="padding-bottom: 3px;"}
*Fig 2: Samples from GLIDE using classifier-free guidance [1].*

Through explicit finetuning, GLIDE can perform image inpainting, allowing specific regions of an image to be modified with additional text input. The model architecture itself is modified to have 
![GLIDE inpainting samples]({{ '/assets/images/team13/GLIDE-inpainting-samples.png' | relative_url }})
*Fig 3: Inpainting samples from GLIDE [1].*

GLIDE begins to struggle when given obscure prompts which describe highly abnormal objects or scenarios. Below are some examples of these failure cases:

![GLIDE failures]({{ '/assets/images/team13/GLIDE_failures.png' | relative_url }})
*Fig 4: Failed samples from GLIDE on atypical prompts [1].*

Another consideration is speed. Unoptimized, GLIDE takes 15 seconds to sample one image on a single A100 GPU. This is much slower compared to other methods like those of GANs, and is thus impractical for real-time applications.

### Code
OpenAI has freely released a smaller version of GLIDE, filtered to remove images of people, to the public for use. The small size of this model hinders its performance and capabilities, but it still offers a taste of what GLIDE (unfiltered) has to offer.

[This](https://colab.research.google.com/github/openai/glide-text2im/blob/main/notebooks/text2im.ipynb) notebook shows how to use GLIDE (filtered) with classifier-free guidance to produce images conditioned on text prompts, and [here](https://github.com/openai/glide-text2im) is the GitHub repository for the filtered version of GLIDE released to the public.



## DALL-E 2
DALL-E 2 (2022) is an improvement to OpenAI's original DALL-E with greater accuracy, diversity, and higher image resolution up to 1024×1024 pixels. The model used is unCLIP, which comes from inverting the image encoder of the CLIP model, hence the name unCLIP.

### Model Architecture
![unCLIP architecture]({{ '/assets/images/team13/unCLIP_architecture.png' | relative_url }})
*Fig 5: High level overview of uncLIP [[6](https://arxiv.org/pdf/2204.06125.pdf)].*

We can visualize the process of unCLIP with the diagram in Figure 5. Above the dotted line is where training on the CLIP model is done to obtain the joint representation space for text and images. Once that is done, the model unCLIP is shown below the dotted line to construct an image based on the given text. There are two main components in unCLIP that we will look at in detail below: the prior and the decoder. <br/> <br/>
Let $$x$$ be the images and $$y$$ be the captions, and we have a training set in the form $$(x, y)$$. Given an image $$x$$, let $$z_i$$ be its CLIP image and $$z_t$$ be its text embeddings. Putting together the prior $$P(z_i|y)$$ and decoder $$P(x|z_i,y)$$ gives us the generative model $$P(x|y)$$ with the equation:

$$P(x|y)=P(x,z_i|y)=P(x|z_i,y)P(z_i|y)$$

#### Prior
The main purpose of the prior is to learn a generative model of the image embeddings from captions $$y$$ to generate CLIP embeddings $$z_i$$. There are two different prior model types: Autoregressive (AR) prior and Diffusion prior. 
* Autoregressive (AR) prior: a CLIP image embedding $$z_i$$ is converted into a sequence of discrete codes and predicted autoregressively conditioned on the caption $$y$$.
* Diffusion prior: $$z_i$$ is directly modeled using a Gaussian diffusion model conditioned on y.

#### Decoder
The decoder is necessary to invert images $$x$$ using their CLIP embeddings $$z_i$$. More specifically, this component projects and adds CLIP embeddings to the existing timestep, and the CLIP embeddings are also separated into four additional tokens of context that are concatenated to the sequence of outputs from the GLIDE text encoder. In addition, two upsampler models are trained and used during this stage to increase resolution to 1024x1024, making unCLIP produce higher quality images than other models.


### Results and Limitations
By explicitly generating image representations, unCLIP improves image diversity with minimal loss in photorealism and caption similarity when compared with GLIDE (Figure 6).

![unCLIP diversity]({{ '/assets/images/team13/unCLIP_diversity.png' | relative_url }})
*Figure 6. Samples with increasing guidance scale coefficient from unCLIP and GLIDE for the prompt "A green vase filled with red roses sitting on top of table" [[6](https://arxiv.org/pdf/2204.06125.pdf)].*

However, unCLIP is worse at binding attributes to objects than a corresponding GLIDE model. For example, as seen in Figure 7, unCLIP struggles to bind two separate objects (cubes) to two separate attributes (color). A potential reason for this is that CLIP embedding itself does not explicitly bind attributes to objects, and so reconstructions from the decoder often mix up attributes and objects
![unCLIP challenges]({{ '/assets/images/team13/unCLIP_challenges.png' | relative_url }})
*Figure 7. Samples from unCLIP and GLIDE for the prompt “a red cube on top of a blue cube” [[6](https://arxiv.org/pdf/2204.06125.pdf)].*

Another struggle unCLIP faces is producing details in some complex scenes (Figure 8). This is likely a result of the decoder hierarchy producing an image at a low base resolution of 64 × 64 and then upsampling it. Training with a higher base resolution may alleviate this problem at the cost of greater computation expenses.
![unCLIP challenges]({{ '/assets/images/team13/unCLIP_complex_scenes.png' | relative_url }})
*Figure 8. Samples from unCLIP for a high quality photo of Times Square [[6](https://arxiv.org/pdf/2204.06125.pdf)].*

### Code
[Here](https://github.com/lucidrains/DALLE2-pytorch) is an implementation of OpenAI's DALL-E 2 written in Pytorch.

To make use of it, begin by installing the package: 
```
$ pip install dalle2-pytorch
```

Training DALLE-2 is a 3 step process which involves training CLIP, a Prior, and a Decoder. We will use the diffusion prior in this example.

First, make the necessary imports:
```
import torch
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, CLIP
```

Next, train CLIP, which is the most important part.
```
clip = CLIP(
    dim_text = 512,
    dim_image = 512,
    dim_latent = 512,
    num_text_tokens = 49408,
    text_enc_depth = 6,
    text_seq_len = 256,
    text_heads = 8,
    visual_enc_depth = 6,
    visual_image_size = 256,
    visual_patch_size = 32,
    visual_heads = 8
).cuda()

# mock data
text = torch.randint(0, 49408, (4, 256)).cuda()
images = torch.randn(4, 3, 256, 256).cuda()

# train
loss = clip(
    text,
    images,
    return_loss = True
)

loss.backward()

# do above for many steps ...
```
Now train the diffusion prior.
```
prior_network = DiffusionPriorNetwork(
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8
).cuda()

diffusion_prior = DiffusionPrior(
    net = prior_network,
    clip = clip,
    timesteps = 1000,
    sample_timesteps = 64,
    cond_drop_prob = 0.2
).cuda()

loss = diffusion_prior(text, images)
loss.backward()

# do above for many steps ...
```
Finally, train the decoder with Unet.
```
unet1 = Unet(
    dim = 128,
    image_embed_dim = 512,
    text_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults=(1, 2, 4, 8),
    cond_on_text_encodings = True    # set to True for any unets that need to be conditioned on text encodings
).cuda()

unet2 = Unet(
    dim = 16,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults = (1, 2, 4, 8, 16)
).cuda()

decoder = Decoder(
    unet = (unet1, unet2),
    image_sizes = (128, 256),
    clip = clip,
    timesteps = 100,
    image_cond_drop_prob = 0.1,
    text_cond_drop_prob = 0.5
).cuda()

for unet_number in (1, 2):
    loss = decoder(images, text = text, unet_number = unet_number) # this can optionally be decoder(images, text) if you wish to condition on the text encodings as well, though it was hinted in the paper it didn't do much
    loss.backward()

# do above for many steps
```
Now that we have a prior and a decoder, we can initialize DALL-E 2 and generate images for a given prompt!
```
dalle2 = DALLE2(
    prior = diffusion_prior,
    decoder = decoder
)

images = dalle2(
    ['cute puppy chasing after a squirrel'],
    cond_scale = 2. # classifier free guidance strength (> 1 would strengthen the condition)
)

# save your image (in this example, of size 256x256)
```
*Code example courtesy of [https://github.com/lucidrains/DALLE2-pytorch](https://github.com/lucidrains/DALLE2-pytorch)*

## References
[1] Nichol, Alex, et al. ["Glide: Towards photorealistic image generation and editing with text-guided diffusion models."](https://arxiv.org/pdf/2112.10741.pdf) arXiv preprint arXiv:2112.10741 (2021).

[2] Dhariwal and Nichol. ["Diffusion Models Beat GANs on Image Synthesis."](https://papers.nips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf) NIPS, 2021.

[3] Ho, et al. ["Denoising diffusion probabilistic models."](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf) NIPS, 2020.

[4] Radford, Alec, et al. ["Learning transferable visual models from natural language supervision."](http://proceedings.mlr.press/v139/radford21a/radford21a.pdf) International conference on machine learning. PMLR, 2021.

[5] Ho and Salimans. ["Classifier-Free Diffusion Guidance."](https://openreview.net/pdf?id=qw8AKxfYbI) NIPS, 2021.

[6] Ramesh, Aditya, et al. ["Hierarchical text-conditional image generation with clip latents."](https://arxiv.org/pdf/2204.06125.pdf) arXiv preprint arXiv:2204.06125 (2022).

---
