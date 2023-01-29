---
layout: post
comments: true
title: Anime GAN Proposal
author: Yuxi Chang
date: 2023-1-28
---


> Generative adversarial network (GAN) is a type of generative nural network capable of varies tasks such as image creation, super-resolution, and image classifications[1]. This project will explore the usage of GAN model on the specific domain of **anime character** and try to find improvemnts on current GAN model such as image quality, style specifications and training optimizations (sample size reductions etc.).


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Project Proposal
![WaifuLab]({{'/assets/images/team09/proposal_waifu_lab.png' | relative_url}}){: style="width: 400px; max-width: 100%;"}
*Fig 1. Sample Output from WaifuLab (GAN)[2].

Recent progress on anime art generation has sparked a sense of excitement in both the AI and anime community. Many AI generated arts have emerged on the internet. As shown in Fig 1, it is an anime character generated using the online tool [WaifuLab](https://waifulabs.com/generate)[2] which relies on a generator model called GAN. GAN (generative adversarial network) is the base model for many current art generator AIs. The project will explore the usage of GAN on anime character generations and try different variations of GAN (possibly other models) to find potential improvements on the style (specifically the hand parts of the art) and resolution of the image output. Since the time of this project is limited, it is also necessary to explore ways of reducing the training time for GAN using smaller sample size.    

## Related Works
- Creswell A., White T., et al. ["Generative Adversarial Networks: An Overview"](https://ieeexplore.ieee.org/abstract/document/8253599/authors#authors)
    - An overview of GAN, helpful for understanding the basic of GAN
- Karras T., Laine S. & Alia T. ["A Style-Based Generator Architecture for Generative Adversarial Networks"](https://arxiv.org/abs/1812.04948)
    - An new architecture for GAN developed by NVIDA that allow control on the style of the generated faces such as eye color and hair color, which is important when creating desired anime characters. StyleGen2 also provide nice results for anime domain. 
    - Code base: https://github.com/NVlabs/stylegan2
- Karras T., Aittala M., et al. ["Training Generative Adversarial Networks with Limited Data"](https://arxiv.org/abs/2006.06676)
    - An introduction to an improved augmentation method that significantly reduces the required taining sample size to achieve similar performance in StyleGan2
    - Code base: https://github.com/NVlabs/stylegan2-ada
    - Pytorch adaptation: https://github.com/NVlabs/stylegan2-ada-pytorch 
- Ruan S. ["Anime Characters Generation with Generative
Adversarial Networks"](https://ieeexplore.ieee.org/abstract/document/9918869/metrics#metrics) 
    - A more specific review of Anime character generation using GAN. The author also provided challenges in the field of anime generation using GAN
- ...more to come while doing the research
---

## Reference

[1] Creswell A., White T., et al. "Generative Adversarial Networks: An Overview" in IEEE Signal Processing Magazine, vol. 35, no. 1, pp. 53-65, Jan. 2018, doi: 10.1109/MSP.2017.2765202.

[2] Liu, R. "Welcome to Waifu Labs v2: How do AIs Create?", Jan, 2022. Retrieved from: https://waifulabs.com/blog/ai-creativity  

---
