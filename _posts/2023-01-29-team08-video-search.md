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

---
