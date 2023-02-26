---
layout: post
comments: true
title: Point-Voxel Diffusion
author: Michael Simon, Victor Lin
date: 2022-01-18
---


> 3D Point Clouds are becoming more and more common as CV models are applied to real world applications like Autonomous Driving. In this blog we explore the cutting edge implementations of 3D Point Cloud Diffusion and its applications in 3D modeling, including generation and completion. Specifically, Point-Voxel Diffusion can assist in clarifying noisy 3D point cloud scans with multiple completion candidates, as well synthesizing realistic smooth shapes.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Three Most Relevant Research Papers

**3D Shape Generation and Completion through Point-Voxel Diffusion**

Proposes a novel model for 3D generative modeling called Point-Voxel Diffusion (PVD) which is capable of both unconditional shape generation and conditional, multi-modal shape completion. [1]

Website article: https://alexzhou907.github.io/pvd

Paper: https://arxiv.org/abs/2104.03670

**PointFlow: 3D Point Cloud Generation with Continuous Normalizing Flows**

Proposes a novel method of generating 3D point clouds via a probabilistic framework to that models them as a distribution of distributions. [2]

Website article: https://www.guandaoyang.com/PointFlow/

Paper: https://arxiv.org/abs/1906.12320

**Learning Gradient Fields for Shape Generation**

Proposes a method for 3D point cloud generation using stochastic gradient ascent on an unnormalized probability density, which moves the sampled points in the direction of high-likelihood regions. [3]

Website article: https://www.cs.cornell.edu/~ruojin/ShapeGF/

Paper: https://arxiv.org/abs/2008.06520

## References

[1] Zhou, Linqi et al. "3D Shape Generation and Completion Through Point-Voxel Diffusion." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).* 2021.

[2] Yang, Guandao et al. "PointFlow: 3D Point Cloud Generation with Continuous Normalizing Flows." (2019).

[3] Cai, R. et al. “Learning gradient fields for shape generation,” *Computer Vision – ECCV* (2020: pp. 364–381.