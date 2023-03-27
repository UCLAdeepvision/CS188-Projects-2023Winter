---
layout: post
comments: true
title: Driving Simulators
author: Victoria Lam, Austin Law
date: 2023-02-26
---

> Self-driving is a hot topic among deep vision learning. One way of training driving models is using imitation learning. In this work, we focus on reproducing the findings from “End-to-end Driving via Conditional Imitation Learning.” To do so, we utilize CARLA, a driving simulator, and emulate the models created in said paper using their provided dataset.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc} 

# Introduction
Our work explores the idea of conditional imitation learning model as experimented by “End-to-end Driving via Conditional Imitation Learning”, which addresses the limitations of imitation learning by proposing to condition imitation learning on high-level command input. This work will also draw ideas from “Conditional Affordance Learning for Driving in Urban Environments,” which also implements a similar method of using high-level directional inputs to improve autonomous driving. Imitation learning has not scaled up to fully autonomous urban driving due to limitations to its learning model. As explained in “End-to-end Driving via Conditional Imitation Learning”, one limitation is the assumption that the optimal action can be inferred from the perceptual input alone, however this does not hold in practice.

One method of dealing with this limitation is conditional imitation learning. We plan to first understand the standard imitation learning model. Then based on the original model, we will make modifications to conduct conditional imitation learning and compare the results from the original model.

# Technical Details

## CARLA Simulator
To explore and experiment with autonomous driving, we will be using CARLA (Car Learning to Act). CARLA is an open-source simulator implemented in Unreal Engine 4 used for autonomous driving research. CARLA has been developed from the ground up to support development, training, and validation of autonomous urban driving systems. CARLA provides open digital assets, such as urban layouts, buildings, vehicles, and pedestrians. The simulation platform supports flexible specification of sensor suites and environmental conditions.

![CARLA Simulator](/assets/images/team07/carla_2.jpg)
![Image Segmentation in CARLA](/assets/images/team07/carla_1.png)

## Imitation Learning Model
Imitation learning’s aim is to train the agent by demonstrating the desired behavior. Imitation learning is a form of supervised learning. The standard imitation learning model maps directly from raw input to control output and is learned from data in an end-to-end fashion. The data for imitation learning is collected through i.e. video footage and input monitoring of a human driver driving in the simulated environment.
The Data
The data is given as two separate but associated "datasets": a set of images and a set of vehicle measurements. While stored separately, they are associated based on the time step (captured at the same time). The image is a 200x88 image of the vehicle's front camera. Measurements include speed, acceleration, position, noise, etc.
The Model
The purpose of the model is to generate the set of actions for a given time step, described by the steering angle and acceleration (represented by magnitude of gas and brake). It takes as input the image and measurements of a time step and is expected to output three values: steering angle, gas amount, and brake amount.

The model has three main sections: a convolutional section and two fully connected sections. The image goes through the convolution section while the measurements are passed through one of the fully connected sections. Afterwards, the two results are concatenated and passed through a final fully connected section until the output is generated.

The convolutional section of the model utilizes convolutional layers to process images from the dataset. The image module consists of 8 convolutional and 2 fully connected layers. The convolution kernel size is 5 in the first layer and 3 in the following layers. The first, third, and fifth convolutional layers have a stride of 2. The first convolutional layer begins with 32 channels and increases to 256 channels in the last layer.

The fully-connected layers each contain 512 units. The measurements are passed through the fully connected sections. The speed of the car is used as the measurement. After all hidden layers, we used ReLU nonlinearities. 20% dropout is used after convolutional layers and 50% dropout is applied after fully-connected hidden layers.

Included below is an image and code snippet of the architecture.

![Imitation Learning Architecture](/assets/images/team07/imitation_learning.png)

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding='valid', dropout=.2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.activation(x)
                            
        return x
    
class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=.5) -> None:
        super().__init__()
        self.ff = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.ff(x)
        x = self.dropout(x)
        x = self.activation(x)

        return x

class ImitationLearningNetwork(nn.Module):
    def __init__(self):
        super(ImitationLearningNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            ConvBlock(3, 32, 5, 2),
            ConvBlock(32, 32, 3, 1)
        )
        self.conv2 = nn.Sequential(
            ConvBlock(32, 64, 3, 2),
            ConvBlock(64, 64, 3, 1)
        )
        self.conv3 = nn.Sequential(
            ConvBlock(64, 128, 3, 2),
            ConvBlock(128, 128, 3, 1)
        )
        self.conv4 = nn.Sequential(
            ConvBlock(128, 256, 3, 1),
            ConvBlock(256, 256, 3, 1)
        )
        self.fc1 = FCBlock(8192, 512)
        self.fc2 = FCBlock(512, 512)

        self.measurements_fc1 = FCBlock(22, 128)
        self.measurements_fc2 = FCBlock(128, 128)

        self.joint_fc = FCBlock(640, 512)

        self.no_branch = nn.Sequential(
            FCBlock(512, 256),
            FCBlock(256, 256),
            nn.Linear(256, 3)
        )

def forward(self, image, measurements):
        x = image
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.fc2(x)

        measurements = self.measurements_fc1(measurements)
        measurements = self.measurements_fc2(measurements)

        """ Joint sensory """
        j = torch.concat([x, measurements], 1)
        j = self.joint_fc(j)

        return self.no_branch(j)
```
## Optimization
All models were trained using the Adam solver with learning rate `lr=.0002`.

The criterion used is MSELoss.

## Results

After 15 epochs of training over 3288 * 200 = 657600 rows of data, our model reached an average validation loss of .0950.
> TODO: Include observations on performance in CARLA

## The Next Step
This is currently the generic imitation learning, and not conditional imitation learning. The next step will be to replace the last layer with branches, which are chosen by the control signal in the measurements.The original paper also followed a similar process, first training a generic imitation learning model and then introducing conditional imitation learning. However, the code and model structure of the generic imitation learning is not included in their paper nor in their code, so parts of the model had to be designed ourselves. Additionally, we would like to use the model directly in the Carla environment and see how it controls a simulated vehicle. As this is also not directly in the code repository, it is another aspect of the project that we will have to figure out ourselves.

## Conditional Imitation Learning
Conditional imitation learning is an extended form of imitation learning. However, there are two primary differences. First, there is a “command” passed for every step, which represents some high level goal of this step. In the paper, they are described as “follow lane”, “left”, “right”, and “straight.” This command is used as part of the measurements in the imitation model; for conditional imitation, though, it is removed from the measurements and used separately. Second, the structure of the model is modified to accommodate for the above change. Instead of a final joint layer, there exist multiple (in this case four) branches, all of which are separate fully connected blocks. Which branch is used is chosen by the command that is provided, and thus the output will change accordingly.

![Conditional Imitation Learning Architecture](/assets/images/team07/conditional_learning.png)
# Issues Encountered
One major issue was difficulty in understanding fully the paper we are referencing. For example, the branching aspect of the model is completely novel, which is one reason we opted to train the generic imitation model first and then expand to the branching version later. However, because the final code is the branching model, we had to recreate parts of the model ourselves. For example, the way the data is parsed and passed to the model had to be recreated from scratch. 

Another difficulty is integrating the model with CARLA. This requires the creation of our own Agent to place in the simulation; while the agent is provided in the repository, how to actually integrate it into the simulation is not and requires its own research.

# References

_End-to-end Driving via Conditional Imitation Learning._
Codevilla, Felipe and Müller, Matthias and López, Antonio and Koltun, Vladlen and Dosovitskiy, Alexey. ICRA 2018. https://arxiv.org/pdf/1710.02410.pdf

_CARLA: An Open Urban Driving Simulator_. Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, Vladlen Koltun; PMLR 78:1-16. https://arxiv.org/pdf/1806.06498.pdf