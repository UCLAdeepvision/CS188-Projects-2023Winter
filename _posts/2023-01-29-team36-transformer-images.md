---
layout: post
comments: true
title: Language Representation for Computer Vision 
author: Thant Zin Oo
date: 2023-01-29
---


> In this blog I will investigate natural language representations as they are used in computer vision. A foray into the predominant language architecture, Transformers, will be linked to tasks in image captioning and art generation.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Relevant Papers
[1] **Attention is All You Need**
- https://arxiv.org/abs/1706.03762
- https://huggingface.co/docs/transformers/index

The seminal paper on the Transformer architecture introduces attentional mechanisms for capturing long range dependencies between language. 
<br/><br/>


[2] **Learning Transferable Visual Models From Natural Language Supervision**
- https://arxiv.org/pdf/2103.00020v1.pdf
- https://github.com/openai/CLIP

OpenAI's paper on CLIP (Connecting Text and Images) demonstrates the zero-shot potential of pretraining on iamge-caption pairs. 
<br/><br/>


[3] **Hierarchical Text-Conditional Image Generation with CLIP Latents**
- https://arxiv.org/abs/2204.06125
- https://huggingface.co/spaces/multimodalart/latentdiffusion

This paper leverages CLIP's text to image embedding capabilities as an encoder, combined with a diffusion based decoder, to generate images conditioned on text prompts. 
