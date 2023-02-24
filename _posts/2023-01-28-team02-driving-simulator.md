---
layout: post
comments: true
title: Driving Simulator
author: Siwei Yuan, Yunqiu Han
date: 2023-01-28
---

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Driving simulators provide the fundamental platform for autonomous driving researches. By integrating various deep learning or reinforcement learning pipelines and engines, driving simulators facilitate and standardize model training, validation, testing, and evaluation. The iterations in driving simulators have aimed at a better integration of powerful DL/RL platforms and realistic scene simulations.

In this paper, we study Metadrive, a lightweight driving simulator supporting compositional map generation and a variety of sensory inputs. We investigate the implementation and configuration of the environment generation, as well as agents' policies, which defines the way they interact with the environment. By gathering FPV camera image and traning customized models, we create new policies using RGB image instead of Lidar data and evaluate the performace of agents in various environments.


## Environment Setup
1. Create a new virtual environment for Metadrive
   ```
   conda create --name Metadrive python=3.8
   conda activate Metadrive
   ```
2. Install metadrive as instructed here: [Metadrive](https://github.com/metadriverse/metadrive)
3. Install relevant packages:
   ```
   conda install pytorch cudatoolkit=11.3 -c pytorch -c nvidia
   conda install -c pytorch torchvision
   ```


## Research Direction
Our goal is to study the effectiveness of supervised learning using RGB image as input, by comparing its accuracy with the RL agent in MetaDrive. We took the following steps to achieve this goal:
1. Obtain RGB images and ground-truth actions from the driving simulation on MetaDrive using the IDM policy.
2. Train a supervised learning model using the RGB images as input and the ground-truth actions as output.
3. Define a policy which uses the trained model to decide the action based on the observation.
4. Evaluate the performance of the policy.



## Environment (Map & Agent) Configuration
The environment configuration is explained in the [Metadrive Documentation](https://metadrive-simulator.readthedocs.io/en/latest/config_system.html). For testing and concept validation purposes, we configured a fundamental environment with one-lane on each direction, and no traffic except for the agent. The block generation is also limited to Straight(S) and Circular(C). The size of RGM camera image is set to 128*128.
```python
config = dict(
        use_render=True,
        offscreen_render=True,
        manual_control=True,
        traffic_density=0,
        environment_num=100,
        random_agent_model=False,
        start_seed=random.randint(0, 1000),
        vehicle_config = dict(image_source="rgb_camera", 
                              rgb_camera= (128 , 128),
                              stack_size=1),
        block_dist_config=PGBlockDistConfig,
        random_lane_width=True,
        random_lane_num=False,
        map_config={
            BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
            BaseMap.GENERATE_CONFIG: "SCCCSCCC",  # it can be a file path / block num / block ID sequence
            BaseMap.LANE_WIDTH: 3.5,
            BaseMap.LANE_NUM: 1,
            "exit_length": 50,
    })
```


## Extracting RGB Image  With Ground Truth Actions From Metadrive
In order to train a model that outputs an action space prediction based on the input RGB image, we first need to collect enough images with ground-truth action labeling. All the data required at this step can be obtained from the *obervation* and *info* returned by *env.step()*. The action space, as defined in the IDM Policy, consists of two numbers, representing the steering and acceleration respectively. The numbers are used to directly name the image file, which is placed in validation set or training set as configured.
```python
o, r, d, info = env.step([0, 0])
action_space = (info['steering'], info['acceleration'])
if PRINT_IMG and i%SAMPLING_INTERVAL == 0:
    img = np.squeeze(o['image'][:,:,:,0]*255).astype(np.uint8)
    img = img[...,::-1].copy()
    img = Image.fromarray(img)
    root_dir = os.path.join(os.getcwd(), 'dataset', 'val')
    img_path = os.path.join(root_dir, str(action_space) + ".png")
    img.save(str(img_path))
```
The image obtained, as shown below, contains all the information about the lane that the agent vehicle requires to make a good prediction. 
<!-- !!!!! ADD IMAGE HERE !!!!! -->
![fig1]({{ '/assets/images/Team02/fig1.png' | relative_url }})
<!-- {: style="width: 128; max-width: 100%;"} -->
*Fig 1. RGB image example*


## Training
We defined a dataloader to handle the dataset we collected, which simply reads in the image file and processes its filename to store as the label. Some code implemented are omitted for readability.
```python
class Metadrive(Dataset):
    def __init__(self, root_dir, split, transform=None):
        for path in os.listdir(os.path.join(root_dir, 'dataset', split)):
          if os.path.isfile(os.path.join(root_dir, 'dataset', split, path)):
            self.filenames.append(path)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, 'dataset', self.split, self.filenames[idx])
        image = Image.open(image_path)
        label = self.filenames[idx][1:-5]
        label = label.split(', ')
        steering = float(label[0])*100
        acceleration = float(label[1])*10
        return image, torch.Tensor([steering, acceleration])
```
In order to validate that our implementation for the data loader is correct, we used a pretrained ResNet18 from PyTorch and added a fully connected linear layer at the end, transforming the 512-dim tensor to the 2-dim tensor, which represents the action space we intend to obtain from the model prediction. A naive training method was adopted, and the model was trained for 20 epochs.
```
Epoch 1/20: 100%|██████████| 15/15 [00:02<00:00,  7.49it/s, loss=13.9]
Validation set: Average loss = 18.0364
Epoch 2/20: 100%|██████████| 15/15 [00:01<00:00, 11.60it/s, loss=10.6]
Validation set: Average loss = 16.7325
Epoch 3/20: 100%|██████████| 15/15 [00:01<00:00,  9.50it/s, loss=10.8]
Validation set: Average loss = 15.1718
......
Epoch 20/20: 100%|██████████| 15/15 [00:01<00:00, 11.27it/s, loss=15.7]
Validation set: Average loss = 14.9256
```
It is worth noticing that before training, we scale up the steering value by 100 and acceleration by 10 so that the values don't appear too small. However, as shown from the above output, the loss didn't decrease noticeably, and the prediction accuracy was not promising. The analysis of this result and possible improvements to be done in the future will be discussed in later sections.



## Defining Our Own Policy
We defined a policy class named `RGBPolicy` which inherits from `BasePolicy` in MetaDrive. Upon initialization, we set `self.model` to be the trained supervised learning model which we load from a previouly saved checkpoint. We overrode the `act()` function, so that it obtains an RGB image from the controlled object, converts the image to tensor, and calls the `self.model` on the tensor to obtain an action (including steering and acceleration). The code is as follows:
```python
class RGBPolicy(BasePolicy):
    PATH = "model.pt"
    
    def __init__(self, control_object, random_seed):
        super(RGBPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.model = Resnet(mode='linear',pretrained=True)
        checkpoint = torch.load(self.PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def act(self, *args, **kwargs):
        # get a PNMImage
        img = self.control_object.image_sensors["rgb_camera"].get_image(self.control_object)

        # PNMImage to tensor
        img = self.__convert_img_to_tensor(img)

        data_transform = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        img = data_transform(img)

        action = self.model(img)[0].detach().numpy()         
        action[0] = action[0]/100
        action[1] = action[1]/10
        return action

    def __convert_img_to_tensor(self, img):
        height = img.get_read_x_size()
        width = img.get_read_y_size()

        img_tensor = torch.zeros((1, 3, height, width))

        for x in range(height):
            for y in range(width):
                img_tensor[0,0,y,x] = img.get_red(x,y)
                img_tensor[0,1,y,x] = img.get_green(x,y)
                img_tensor[0,2,y,x] = img.get_blue(x,y)
        
        return img_tensor
```
## Visualization and Evaluation
We visualized our `RGBPolicy` in the MetaDrive environment to see how it performs.

![fig1]({{ '/assets/images/Team02/demo.gif' | relative_url }})


## Future Plan and Possible Areas to Improve
1. Dicrete classification  
Currently, the naive implementation of the model predicts continuous steering and accleration values, and the loss function uses MSE loss. Accurately predicting continuous values can be a difficult task for a pretrained model. Thus, we can categorize the lane information into several discrete situations, e.g. straight, slight left curved, and etc. For each category, we will obtain the ground truth value for steering angle by taking the mean of all values in this specific situation. 
2. Incorporating speed information  
While the action space for the original IDM policy only consists of two numbers, speed is an important value to consider when predicting vehicle movement in autonomous driving. It's reasonable to believe that speed and acceleration are correlated. However, the currect approach does not take speed information into account. Therefore, after classifying the lane category, speed information can be feed into a separate model to predict the acceleration and possibily steering.
3. Fine-tune model  
We will try different pretrained models and finetune them to explore the most suitable model for this task.
4. More data & data augmentation
The current dataset we constructed only contains ~1000 images for training and ~200 images for validation. This is not enough for certain models such as ResNet or VIT. We will try data augmentation to generate more images for training and preventing overfitting.
5. Integrating image segmentation
Image segmentation models can be applied in this task. Before predicting the action space, we can first feed the image to the segmentation model, which may be able to extract focused and useful information specifically about the road or lane.
6. Extend to more complicated scenarios
If driving in a one-lane no traffic scenario is successfully solved, we will extend the project to more realistic scenarios. 


## Possible project topics
1. Integration of an existing algorithm/model with another driving simulator.
2. Evaluation of the effect of sensory inputs removal/addition on an implemented model.
3. Evaluate a trained & tested model under new scenes with certain events.

<!-- Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md) -->

<!-- ## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.
You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].
Please cite the image if it is taken from other people's work. -->

<!-- 
### Table
Here is an example for creating tables, including alignment syntax.
|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          | -->



<!-- ### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
``` -->


<!-- ### Formula
Please use latex to generate formulas, such as:
$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$
or you can write in-text formula $$y = wx + b$$.
### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/). -->

## Reference

[1] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, “CARLA: An Open Urban Driving Simulator,” in Proceedings of the 1st Annual Conference on Robot Learning, 2017, pp. 1–16.  
[2] S. Shah, D. Dey, C. Lovett, and A. Kapoor, “AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles,” 2017. [Online]. Available: https://arxiv.org/abs/1705.05065  
[3] Q. Li, Z. Peng, L. Feng, Q. Zhang, Z. Xue, and B. Zhou, "MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning," arXiv, 2021. doi: 10.48550/ARXIV.2109.12674.  
---



## Code Repository

[CARLA](https://github.com/carla-simulator/carla)

[AirSim](https://github.com/microsoft/AirSim)

[MetaDrive](https://github.com/metadriverse/metadrive)