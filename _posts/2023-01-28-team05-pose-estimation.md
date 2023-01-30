---
layout: post
comments: true
title: Pose Estimation
author: Aristotle Henderson, John Houser
date: 2023-01-28
---

> This post is an investigation and review of pose estimation research and implementation.

<!--more-->

{: class="table-of-content"}

- TOC
    {:toc}

## Three Most Relevant Research Papers

**OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields**
<https://arxiv.org/pdf/1812.08008.pdf> [1]

This paper discusses a method of generating 2D pose estimations in real time for people in an image using trained nonparametric representations.

Code: <https://github.com/CMU-Perceptual-Computing-Lab/openpose>

**Deep High-Resolution Representation Learning for Human Pose Estimation**
<https://arxiv.org/pdf/1902.09212v1.pdf> [2]

This paper describes a high-resolution neural network that connects multiple subnetworks of differing qualities together for high-performance pose tracking.

Code: <https://github.com/leoxiaobin/deep-high-resolution-net.pytorch>

**DensePose: Dense Human Pose Estimation In The Wild**
<https://arxiv.org/pdf/1802.00434v1.pdf> [3]

This paper proposes a model that uses dense representations of the human body with convolutional neural networks to map every pixel to a person's pose.

Code: <https://github.com/facebookresearch/detectron2>

## References

[1] Cao, Zhe, et al. "OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields." _IEEE Transactions on Pattern Analysis and Machine Intelligence_. 2019.

[2] Sun, Ke, et al. "Deep High-Resolution Representation Learning for Human Pose Estimation." _Conference on Computer Vision and Pattern Recognition_. 2019.

[3] Güler, Rıza Alp, et al. "DensePose: Dense Human Pose Estimation In The Wild." _Conference on Computer Vision and Pattern Recognition_. 2018.
