---
layout: post
comments: true
title: Video Grounding
author: Enoch Xu, JR Bronkar (Team 08)
date: 2023-01-29
---


> Topic: Video Search

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}
# Proposal


## Video Search
We are going to work on video grounding for our CS188 project. Though video grounding references the problem of retrieving a specific moment from an untrimmed video by giving an input sentence describing the moment, we are planning on implementing an intelligent video search that can ‘recommend’ or ‘find’ the video(s) associated with a given query sentence (given one query sentence and a video repository, find the video associated with the query). If this is too much to tackle, we may just seek to improve/implement our own video grounding algorithm (given one query sentence and a video, find a moment in the video that is associated with the query sentence).

This will involve both NLP and CV processing and depending on the time involved, we will try to make it performant as well, though accuracy and semantic relevance is the main focus for the project.


### Three Relevant Papers
- 'Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language' [Paper](https://arxiv.org/abs/1912.03590) [1]
- 'Unsupervised Temporal Video Grounding with Deep Semantic Clustering' [Paper](https://arxiv.org/abs/2201.05307) [2]
- 'TubeDETR: Spatio-Temporal Video Grounding with Transformers' [Paper](https://arxiv.org/abs/2203.16434) [3]

## Reference

[1] Zhang, Songyang, et al. “Learning 2d Temporal Adjacent Networks for Moment Localization with Natural Language.” ArXiv.org, 26 Dec. 2020, https://arxiv.org/abs/1912.03590. 

[2] Liu, Daizong, et al. “Unsupervised Temporal Video Grounding with Deep Semantic Clustering.” ArXiv.org, 14 Jan. 2022, https://arxiv.org/abs/2201.05307.

[3] Yang, Antoine, et al. “Tubedetr: Spatio-Temporal Video Grounding with Transformers.” ArXiv.org, 9 June 2022, https://arxiv.org/abs/2203.16434.

# Week 7 Update
## Introduction to video grounding:

Video grounding is the task of grounding natural language descriptions to the visual and temporal features of a video. The requirement to tie together and reason through the connections between these 3 domains makes this task highly complex. However, it proves very useful in applications such as video search, video captioning, and video summarization. We include a basic example of labeling a video clip with the slowfast_r50 model in the main.ipynb jupyter notebook.

## This project:

We started by looking into existing video grounding research. We found the paper [https://arxiv.org/abs/1912.03590] particularly interesting because it proposed a unique way to tackle the task of answering queries with segments of video, also known as moment localization. Long videos may demonstrate many different scenes, and it is very helpful to query and narrow down specific segments within these videos.

So, we decided to take a deeper look into the 2D-TAN code base.

## Current Steps:

We began by setting up and initializing the 2D-TAN baseline model with ActivityNet videos/captions as our data for evaluation. We chose ActivityNet because it was a common data source across papers and was a helpful benchmark for evaluation. Downloading ActivityNet took a lot of work, because of the space requirements we had to add an additional disk.

Setting up and initializing the 2D-TAN model was painful. Our single NVIDIA-K80 google cloud instance struggled to allocate enough resources to the model and we noticed that training a single epoch would require large amounts of time. We realized that given the complexity of models like 2D-TAN, video grounding is an extremely computationally demanding task (add some screenshots). Not only did we run into issues with GPU restraints, but also pytorch requires substantial amounts of RAM when initializing and causes our processes to crash frequently.

## Experiments we plan to do and why:

We plan to conduct a few experiments.

1\. General performance and efficiency differences between using pooling vs convolutional layers in 2D-TAN model. Stacked convolution and pooling have been applied for extracting moment features in previous works and we want to compare the performance differences on ActivityNet.

2\. Another experiment we might hope to run might be to introduce a subset of a dataset coming from a different context than ActivityNet and evaluate the performance on the dataset based on pooling vs convolutional layers and also compare the performance vs ActivityNet. In this way we may see how generalizable the trained model is on a different dataset outside of ActivityNet and also if pooling/convolutional layers affect the generalizability of the model.

3\. (Stretch Goal) Fine-tune baseline model. Once we have introduced a dataset that is different than ActivityNet, we can fine-tune our pre-trained model on the other dataset then see if this leads to performance gains on that dataset and also on our earlier dataset, ActivityNet. This can show us whether our fine-tuning allowed the model to improve and become more generalizable or whether it over-corrected the parameters towards a different dataset.

---
