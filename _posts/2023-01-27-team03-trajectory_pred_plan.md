---
layout: post
comments: true
title: Trajectory Prediction and Planning
author: Weizhen Wang, Baiting Zhu
date: 2023-01-27
---


## Introduction
In recent years, the computer vision community has became more involved in automonomous vehicle. With evergrowing hardware support on modeling more complicated interactions between agents and street objects, trajectory prediction of traffic flows is yielding more promising results. The introduction of Transformer technique also galvanized the deep learning community, and our team will explore the application of Transformer in trajectory prediction.




## Paper Summaries
**TrafficGen:**
This paper first trains a generator from the Waymo data to generate longer training data. In the trajectory prediction task, TrafficGen uses vectorization and Multi-Context Gating (MCG) as the encoder. To decode the information, TrafficGen first uses MLP layers to decide the region of the vehicle, then uses log-normal distribution to generate the vehicle features (i.e. position in region, speed, etc.). After all vehicle are generated, TrafficGen uses the Multipath++ to predict future trajectory. To scale prediction to a longer horizon, TrafficGen only uses global features as input for trajectory prediction.


**Multipath++**
Multipath++ first transform the road features into polylines and agent history as a sequence of states encoded by LSTM. It uses a Gaussian Mixture Model as the prior to preserve the multi-modality of trajectory prediction task. Multipath++ proposes the MCG architecture, which is similar to cross-attention by allowing vehicle to communicate with road elements and vice versa. Finally, Multipath++ sets anchor in the latent space to guide the vehicles.

**LaneGCN**
LaneGCN is one of the earlier works that proposes using vectorized input instead of pixel input. Different from the VectorNet paper, LaneGCN extends graph convolutions with multiple adjacency matrices and along-lane dilation and only uses sparse connection between elements.





## Reference

[1] Feng, Lan, et al. "TrafficGen: Learning to Generate Diverse and Realistic Traffic Scenarios." ArXiv, 2022,  https://doi.org/10.48550/arXiv.2210.06609. Accessed 29 Jan. 2023.

[2] Konev, Stepan. "MPA: MultiPath++ Based Architecture for Motion Prediction." ArXiv, 2022,  https://doi.org/10.48550/arXiv.2206.10041. Accessed 29 Jan. 2023.

[3]Liang, Ming, et al. "Learning Lane Graph Representations for Motion Forecasting." ArXiv, 2020,  https://doi.org/10.48550/arXiv.2007.13732. Accessed 29 Jan. 2023.

[4] Shi, Shaoshuai, et al. "Motion Transformer with Global Intention Localization and Local Movement Refinement." ArXiv, 2022,  https://doi.org/10.48550/arXiv.2209.13508. Accessed 29 Jan. 2023.

## Code Bases
[1] TrafficGen: https://github.com/metadriverse/trafficgen
[2] MultiPath++: https://github.com/stepankonev/waymo-motion-prediction-challenge-2022-multipath-plus-plus
[3] LaneGCN: https://github.com/uber-research/LaneGCN

---