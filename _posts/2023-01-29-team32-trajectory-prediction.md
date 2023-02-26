---

layout: post
comments: true
title: Trajectory Prediction
author: Team 32 (Kevin Jiang, Michael Yang)
date: 2023-01-29

---

> Trajectory prediction in the context of an autonomous vehicle involves predicting how nearby vehicles, pedestrians, and other subjects will move in a real-world environment, which in turn is needed for the autonomous vehicle to maneuver in an optimal and safe manner.

<!--more-->

## Articles and Repositories

1) Stepwise Goal-Driven Networks for Trajectory Prediction
    - Paper: https://doi.org/10.48550/arXiv.2103.14107 [1]
    - Repository: https://github.com/ChuhuaW/SGNet.pytorch [2]
    - This paper introduces a recurrent neural network (RNN) called Stepwise Goal-Driven Network (SGNet) for predicting trajectories of observed agents (e.g. cars and pedestrians).
    - Unlike previous research which model an agent as having a single, long-term goal, SGNet draws on research in psychology and cognitive science to model an agent as having a single, long-term _intention_ that involves a series of goals over time.
    - To this end, SGNet estimates and uses goals at multiple time scales to predict agents' trajectories. It comprises an encoder that captures historical information, a stepwise goal estimator that predicts successive goals into the future, and a decoder to predict future trajectory.
2) GRIP++: Enhanced Graph-based Interaction-aware Trajectory Prediction for Autonomous Driving
    - Paper: https://doi.org/10.48550/arXiv.1907.07792 [3]
    - Repository: https://github.com/xincoder/GRIP [4]
    - This paper introduces an improvement on Graph-based Interaction-aware Trajectory Prediction (GRIP), called GRIP++, to handle both highway and urban scenarios.
    - Specifically, while GRIP performed well for highway traffic, urban traffic is much more complex, involving diverse agents with varying motion patterns and whose behavior affect one another. In addition, GRIP used a fixed graph to represent the relationships between agents, leading to potential performance degradation for urban traffic.
    - GRIP++ addresses these limitations by employing both fixed and dynamic graphs to represent the interactions between many different kinds of agents and predict trajectories for all traffic agents simultaneously.
3) Convolutional Social Pooling for Vehicle Trajectory Prediction
    - Paper: https://doi.org/10.48550/arXiv.1805.06771 [5]
    - Repository: https://github.com/nachiket92/conv-social-pooling [6]
    - This paper introduces a long short-term memory (LSTM) encoder-decoder model that learns interdependencies in vehicle motion and predicts future vehicle trajectories in terms of maneuver classes.
    - This model uses a new technique called _convolutional social pooling_, which involves applying convolutional and max-pooling layers to LSTM social tensors, to encode the historical motion of neighboring vehicles.
    - The model also exploits structure in vehicle motion and lane structure on highways to bin future trajectories into six maneuvers (e.g. lane change to the left, brake in current lane, etc.) and to model interaction between vehicles.

## References

[1] Wang, Chuhua, et al. "Stepwise Goal-Driven Networks for Trajectory Prediction". *ArXiv*, IEEE, 27 Mar 2022, [https://doi.org/10.48550/arXiv.2103.14107](https://doi.org/10.48550/arXiv.2103.14107). *Papers with Code*, Papers with Code, 25 Mar 2021, [www.paperswithcode.com/paper/stepwise-goal-driven-networks-for-trajectory](https://paperswithcode.com/paper/stepwise-goal-driven-networks-for-trajectory), accessed 29 Jan 2023.

[2] Wang, Chuhua and Mingze Xu. "SGNet.pytorch." *GitHub*, GitHub, [www.github.com/ChuhuaW/SGNet.pytorch](https://github.com/ChuhuaW/SGNet.pytorch). *Papers with Code*, Papers with Code, 25 Mar 2021, [www.paperswithcode.com/paper/stepwise-goal-driven-networks-for-trajectory](https://paperswithcode.com/paper/stepwise-goal-driven-networks-for-trajectory), accessed 29 Jan 2023.

[3] Li, Xin, et al. "GRIP++: Enhanced Graph-based Interaction-aware Trajectory Prediction for Autonomous Driving." *ArXiv*, ArXiv, 19 May 2020, [https://doi.org/10.48550/arXiv.1907.07792](https://doi.org/10.48550/arXiv.1907.07792). *Papers with Code*, Papers with Code, [www.paperswithcode.com/paper/grip-graph-based-interaction-aware-trajectory](https://paperswithcode.com/paper/grip-graph-based-interaction-aware-trajectory), accessed 29 Jan 2023.

[4] Li, Xin. "GRIP." *GitHub*, GitHub, [www.github.com/xincoder/GRIP](https://github.com/xincoder/GRIP). *Papers with Code*, Papers with Code, [www.paperswithcode.com/paper/grip-graph-based-interaction-aware-trajectory](https://paperswithcode.com/paper/grip-graph-based-interaction-aware-trajectory), accessed 29 Jan 2023.

[5] Deo, Nachiket and Mohan M. Trivedi. "Convolutional Social Pooling for Vehicle Trajectory Prediction." *ArXiv*, ArXiv, 15 May 2018, [https://doi.org/10.48550/arXiv.1805.06771](https://doi.org/10.48550/arXiv.1805.06771). *Papers with Code*, Papers with Code, 15 May 2018, [www.paperswithcode.com/paper/convolutional-social-pooling-for-vehicle](https://paperswithcode.com/paper/convolutional-social-pooling-for-vehicle), accessed 29 Jan 2023.

[6] Deo, Nachiket and Artem Fedoskin. "conv-social-pooling." *GitHub*, GitHub, [www.github.com/nachiket92/conv-social-pooling](https://github.com/nachiket92/conv-social-pooling). *Papers with Code*, Papers with Code, 15 May 2018, [www.paperswithcode.com/paper/convolutional-social-pooling-for-vehicle](https://paperswithcode.com/paper/convolutional-social-pooling-for-vehicle), accessed 29 Jan 2023.

---