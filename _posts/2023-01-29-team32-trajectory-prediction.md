---

layout: post
comments: true
title: Trajectory Prediction
author: Team 32 (Kevin Jiang, Michael Yang)
date: 2023-02-26

---

> Trajectory prediction in the context of an autonomous vehicle involves predicting how nearby vehicles, pedestrians, and other subjects will move in a real-time environment, which in turn is needed for the autonomous vehicle to maneuver in an optimal and safe manner.

<!--more-->

## Models

### Stepwise Goal-Driven Networks for Trajectory Prediction

Paper: [https://doi.org/10.48550/arXiv.2103.14107](https://doi.org/10.48550/arXiv.2103.14107) [1]

Repository: [https://github.com/ChuhuaW/SGNet.pytorch](https://github.com/ChuhuaW/SGNet.pytorch) [2]

#### Overview

This paper introduces a recurrent neural network (RNN) called Stepwise Goal-Driven Network (SGNet) for predicting trajectories of observed agents (e.g. cars and pedestrians) [1].

Unlike previous research which model an agent as having a single, long-term goal, SGNet draws on research in psychology and cognitive science to model an agent as having a single, long-term *intention* that involves a series of goals over time [1].

To this end, SGNet estimates and uses goals at multiple time scales to predict agents' trajectories. It comprises an encoder that captures historical information, a stepwise goal estimator that predicts successive goals into the future, and a decoder to predict future trajectory [1].

### GRIP++: Enhanced Graph-based Interaction-aware Trajectory Prediction for Autonomous Driving

Paper: [https://doi.org/10.48550/arXiv.1907.07792](https://doi.org/10.48550/arXiv.1907.07792) [3]

Repository: [https://github.com/xincoder/GRIP](https://github.com/xincoder/GRIP) [4]

#### Overview

This paper introduces an improvement on Graph-based Interaction-aware Trajectory Prediction (GRIP), called GRIP++, to handle both highway and urban scenarios [3].

Specifically, while GRIP performed well for highway traffic, urban traffic is much more complex, involving diverse agents with varying motion patterns and whose behavior affect one another. In addition, GRIP used a fixed graph to represent the relationships between agents, leading to potential performance degradation for urban traffic [3].

GRIP++ addresses these limitations by employing both fixed and dynamic graphs to represent the interactions between many different kinds of agents and predict trajectories for all traffic agents simultaneously [3].

### Multimodal Trajectory Prediction Conditioned on Lane-Graph Traversals

Paper: [https://doi.org/10.48550/arXiv.2106.15004](https://doi.org/10.48550/arXiv.2106.15004) [5]

Repository: [https://github.com/nachiket92/PGP](https://github.com/nachiket92/PGP) [6]

#### Overview

This paper introduces Prediction via Graph-Based Policy (PGP), an improvement on multimodal regression for trajectory prediction [5].

While standard multimodal regression aggregates the entirety of the graph representation of the map, the new method instead takes a subset of graph data that is more relevant to the vehicle and selectively aggregates this data [5].

In addition to lateral motion (acceleration, braking, etc.), the model also examines variations in longitudinal motion such as lane changes, utilizing a latent variable [5].

The model utilizes a graph encoder to encode node and agent features, a policy header to determine probable routes, and a trajectory decoder to determine a likely trajectory from this data [5].

#### Technical Details

!["PGP Model [5]"](../assets/images/team32/PGPModel.png)

The model consists of a graph encoder, policy header, and trajectory decoder [5].

The graph encoder utilizes three GRU encoders to encode the nodes, agents, and motion [5]. An example of such an encoder (lines 43-45 of ```pgp_encoder.py``` in the source) is as follows: [6]

```
# Node encoders
self.node_emb = nn.Linear(args['node_feat_size'], args['node_emb_size'])
self.node_encoder = nn.GRU(args['node_emb_size'], args['node_enc_size'], batch_first=True)
```

Since agents respond to other nearby agents in traffic, node encodings are updated using information from agent encodings within a certain distance, utilizing multi-head attention to do so [5]; this is implemented in lines 51-56 of ```pgp_encoder.py``` in the source as: [6]

```
# Agent-node attention
self.query_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
self.key_emb = nn.Linear(args['nbr_enc_size'], args['node_enc_size'])
self.val_emb = nn.Linear(args['nbr_enc_size'], args['node_enc_size'])
self.a_n_att = nn.MultiheadAttention(args['node_enc_size'], num_heads=1)
self.mix = nn.Linear(args['node_enc_size']*2, args['node_enc_size'])
```

The attention output is concatenated with the original encoding and a GNN is then used to aggregate node context [5]. The GNN is implemented using GAT in lines 222-251 of ```pgp_encoder.py``` in the source as: [6]

```
class GAT(nn.Module):
    """
    GAT layer for aggregating local context at each lane node. Uses scaled dot product attention using pytorch's
    multihead attention module.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize GAT layer.
        :param in_channels: size of node encodings
        :param out_channels: size of aggregated node encodings
        """
        super().__init__()
        self.query_emb = nn.Linear(in_channels, out_channels)
        self.key_emb = nn.Linear(in_channels, out_channels)
        self.val_emb = nn.Linear(in_channels, out_channels)
        self.att = nn.MultiheadAttention(out_channels, 1)

    def forward(self, node_encodings, adj_mat):
        """
        Forward pass for GAT layer
        :param node_encodings: Tensor of node encodings, shape [batch_size, max_nodes, node_enc_size]
        :param adj_mat: Bool tensor, adjacency matrix for edges, shape [batch_size, max_nodes, max_nodes]
        :return:
        """
        queries = self.query_emb(node_encodings.permute(1, 0, 2))
        keys = self.key_emb(node_encodings.permute(1, 0, 2))
        vals = self.val_emb(node_encodings.permute(1, 0, 2))
        att_op, _ = self.att(queries, keys, vals, attn_mask=~adj_mat)

        return att_op.permute(1, 0, 2)
```

With the node encoding completed, the data will then pass through the policy header, which utilizes an MLP to output probabilities for each node's edges [5]; this is implemented in lines 37-45 of ```pgp.py``` in the source as: [6]

```
# Policy header
self.pi_h1 = nn.Linear(2 * args['node_enc_size'] + args['target_agent_enc_size'] + 2, args['pi_h1_size'])
self.pi_h2 = nn.Linear(args['pi_h1_size'], args['pi_h2_size'])
self.pi_op = nn.Linear(args['pi_h2_size'], 1)
self.pi_h1_goal = nn.Linear(args['node_enc_size'] + args['target_agent_enc_size'], args['pi_h1_size'])
self.pi_h2_goal = nn.Linear(args['pi_h1_size'], args['pi_h2_size'])
self.pi_op_goal = nn.Linear(args['pi_h2_size'], 1)
self.leaky_relu = nn.LeakyReLU()
self.log_softmax = nn.LogSoftmax(dim=2)
```

Loss for the policy header is calculated using negative log likelihood [5], implemented in ```pi_bc.py``` in the source [6].

Once possible future routes are determined, multi-head attention is utilized again for selective aggregation of context to form possible trajectories and the motion that will be undertaken (which may vary for a given trajectory) [5]. The aggregator is implemented in lines 51-56 of ```pgp.py``` in the source as: [6]

```
# Attention based aggregator
self.pos_enc = PositionalEncoding1D(args['node_enc_size'])
self.query_emb = nn.Linear(args['target_agent_enc_size'], args['emb_size'])
self.key_emb = nn.Linear(args['node_enc_size'], args['emb_size'])
self.val_emb = nn.Linear(args['node_enc_size'], args['emb_size'])
self.mha = nn.MultiheadAttention(args['emb_size'], args['num_heads'])
```

Finally, the decoder utilizes an MLP and K-means clustering to output predicted trajectories [5], with the implementation of the MLP in lines 33-35 of ```lvm.py``` in the source as: [6]

```
self.hidden = nn.Linear(args['encoding_size'] + args['lv_dim'], args['hidden_size'])
self.op_traj = nn.Linear(args['hidden_size'], args['op_len'] * 2)
self.leaky_relu = nn.LeakyReLU()
```

Loss for the decoder is calculated using average displacement error [5], implemented in ```min_ade.py``` in the source [6]. The total loss is the sum of the losses of the policy header and the decoder [5].

## Studies

[TODO] Describe studies we intend to run.

## References

[1] Wang, Chuhua, et al. "Stepwise Goal-Driven Networks for Trajectory Prediction". *ArXiv*, IEEE, 27 Mar 2022, [https://doi.org/10.48550/arXiv.2103.14107](https://doi.org/10.48550/arXiv.2103.14107). *Papers with Code*, Papers with Code, 25 Mar 2021, [www.paperswithcode.com/paper/stepwise-goal-driven-networks-for-trajectory](https://paperswithcode.com/paper/stepwise-goal-driven-networks-for-trajectory), accessed 29 Jan 2023.

[2] Wang, Chuhua and Mingze Xu. "SGNet.pytorch." *GitHub*, GitHub, [www.github.com/ChuhuaW/SGNet.pytorch](https://github.com/ChuhuaW/SGNet.pytorch). *Papers with Code*, Papers with Code, 25 Mar 2021, [www.paperswithcode.com/paper/stepwise-goal-driven-networks-for-trajectory](https://paperswithcode.com/paper/stepwise-goal-driven-networks-for-trajectory), accessed 29 Jan 2023.

[3] Li, Xin, et al. "GRIP++: Enhanced Graph-based Interaction-aware Trajectory Prediction for Autonomous Driving." *ArXiv*, ArXiv, 19 May 2020, [https://doi.org/10.48550/arXiv.1907.07792](https://doi.org/10.48550/arXiv.1907.07792). *Papers with Code*, Papers with Code, [www.paperswithcode.com/paper/grip-graph-based-interaction-aware-trajectory](https://paperswithcode.com/paper/grip-graph-based-interaction-aware-trajectory), accessed 29 Jan 2023.

[4] Li, Xin. "GRIP." *GitHub*, GitHub, [www.github.com/xincoder/GRIP](https://github.com/xincoder/GRIP). *Papers with Code*, Papers with Code, [www.paperswithcode.com/paper/grip-graph-based-interaction-aware-trajectory](https://paperswithcode.com/paper/grip-graph-based-interaction-aware-trajectory), accessed 29 Jan 2023.

[5] Deo, Nachiket, et al. "Multimodal Trajectory Prediction Conditioned on Lane-Graph Traversals." *ArXiv*, ArXiv, 15 Sep 2021, [www.doi.org/10.48550/arXiv.2106.15004](https://doi.org/10.48550/arXiv.2106.15004). *Papers with Code*, Papers with Code, 28 Jun 2021, [www.paperswithcode.com/paper/multimodal-trajectory-prediction-conditioned](https://paperswithcode.com/paper/multimodal-trajectory-prediction-conditioned), accessed 26 Feb 2023.

[6] Deo, Nachiket. "PGP." *GitHub*, GitHub, [www.github.com/nachiket92/PGP](https://github.com/nachiket92/PGP). *Papers with Code*, Papers with Code, 28 Jun 2021, [www.paperswithcode.com/paper/multimodal-trajectory-prediction-conditioned](https://paperswithcode.com/paper/multimodal-trajectory-prediction-conditioned), accessed 26 Feb 2023.

---
