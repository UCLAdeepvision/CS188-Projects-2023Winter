---
layout: post
comments: true
title: Driving Simulator
author: Siwei Yuan, Yunqiu Han
date: 2023-01-28
---

>Driving simulators provide the fundamental platform for autonomous driving researches. In this blog, we explores Metadrive, explains the technical details of implementing a purely visual based policy, and evalutes its performance.


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
The resulting generated evironment is shown below:
![fig1]({{ '/assets/images/Team02/env.png' | relative_url }})
*Fig 1. Generated Environment*

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
![fig2]({{ '/assets/images/Team02/fig1.png' | relative_url }})
<!-- {: style="width: 128; max-width: 100%;"} -->
*Fig 2. RGB Image Example*


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
        steering = (float(label[0])-steering_mean)/steering_std
        acceleration = (float(label[1])-accel_mean)/accel_std
        return image, torch.Tensor([steering, acceleration])
```
In order to validate that our implementation for the data loader is correct, we used a pretrained ResNet18 from PyTorch and added a fully connected linear layer at the end, transforming the 512-dim tensor to the 2-dim tensor, which represents the action space we intend to obtain from the model prediction. We set different learning rates for different layers, and we used MSE loss as the criterion since we are dealing with a continuous value prediction model.
```python 
optimizer = torch.optim.SGD(model.resnet.fc.parameters(), lr=0.01, momentum=0.9)
for name, param in model.named_parameters():
    if param.requires_grad and 'fc' not in name:
        optimizer.add_param_group({'params': param, 'lr':0.005})

criterion = nn.MSELoss()
```
```
Epoch 1/10: 100%|██████████| 15/15 [00:01<00:00, 10.91it/s, loss=5.09e+3]
Validation set: Average loss = 12432554169393362758105300992.0000
Epoch 2/10: 100%|██████████| 15/15 [00:01<00:00, 11.38it/s, loss=1.8e+3]
Validation set: Average loss = 5299107328.0000
Epoch 3/10: 100%|██████████| 15/15 [00:01<00:00, 10.88it/s, loss=11]
Validation set: Average loss = 206780.1484
Epoch 4/10: 100%|██████████| 15/15 [00:01<00:00, 10.42it/s, loss=1.43]
Validation set: Average loss = 6.6199
Epoch 5/10: 100%|██████████| 15/15 [00:02<00:00,  6.32it/s, loss=0.873]
Validation set: Average loss = 1.2147
Epoch 6/10: 100%|██████████| 15/15 [00:02<00:00,  7.08it/s, loss=1.16]
Validation set: Average loss = 1.0740
......
Epoch 10/10: 100%|██████████| 15/15 [00:01<00:00, 10.67it/s, loss=0.769]
Validation set: Average loss = 0.9185
```
It is worth noticing that before training, we standardize the values of steering angle and acceleration (code shown below). As shown from the above output, the loss decreased noticeably, but the best loss we could obtain stayed at around 0.9 without further improving. The analysis of this result and possible improvements to be done in the future will be discussed in later sections.
```python
def get_mean_std(root_dir):
    steering = []
    accel = []

    for path in os.listdir(os.path.join(root_dir, 'dataset', 'train')):
        if os.path.isfile(os.path.join(root_dir, 'dataset', 'train', path)):
            label = path[1:-5]
            label = label.split(', ')
            steering.append(float(label[0]))
            accel.append(float(label[1]))

    steering = np.array(steering)
    accel = np.array(accel)

    return np.mean(steering), np.std(steering), np.mean(accel), np.std(accel)
```



## Defining Our Own Policy
We defined a policy class named `RGBPolicy` which inherits from `BasePolicy` in MetaDrive. Upon initialization, we set `self.model` to be the trained supervised learning model which we load from a previouly saved checkpoint. We overrode the `act()` function, so that it obtains an RGB image from the controlled object, converts the image to tensor, and calls the `self.model` on the tensor to obtain an action (including steering and acceleration). The code is as follows:
```python
class RGBPolicy(BasePolicy):
    PATH = "model.pt"
    
    def __init__(self, control_object, random_seed):
        super(RGBPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.model = Resnet()
        self.model.load_state_dict(torch.load(RGBPolicy.PATH, map_location=torch.device('cpu')))
        self.model.eval()

    def act(self, *args, **kwargs):
        # get a PNMImage
        img = self.control_object.image_sensors["rgb_camera"].get_image(self.control_object)

        # PNMImage to tensor
        img = self.__convert_img_to_tensor(img)

        data_transform = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        img = data_transform(img)

        action = self.model(img)[0].detach().numpy()
        action[0] = action[0]*0.06545050570131895 + 0.005975310273351187
        action[0] *= 2.3
        action[1] = action[1]*0.37149717438120655 + 0.3121460530513671
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
It is worth noticing that in the above code, we have to unnormalize the values of steering angle and acceleration returned by the model. And we applied a scaling factor to the steering, which is a value determined by emperical trials.


## Visualization and Evaluation
We visualized our `RGBPolicy` in the MetaDrive environment to see how it performs. The two gifs are from different trials.

![fig3]({{ '/assets/images/Team02/demo.gif' | relative_url }})
*Fig 3. Vehicle's FPV View*
![fig4]({{ '/assets/images/Team02/demo3.gif' | relative_url }})
*Fig 4. Vehicle In The Env*

As the gifs show, the vehicle is able to follow the track in general, but it fails to perfectly avoid the white lane and the center yellow lane. This can be due to the model failing to properly predict a larger steering value when the vehicle approaches lane edges on curved tracks. Since the backbone model used is a pretrained ResNet which is good at doing categorization tasks, it is possible that it outputs values according to the type of situation, while not taking the trivial position difference of the vehicle into consideration.

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
If driving in a one-lane no traffic scenario is successfully solved, we will extend the project to more realistic scenarios. Starting from the introduction of traffic into the scene, we will gradually move towards incoporating double lanes and even traffic blocks.

## Reference
[1] Q. Li, Z. Peng, L. Feng, Q. Zhang, Z. Xue, and B. Zhou, "MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning," arXiv, 2021. doi: 10.48550/ARXIV.2109.12674.  
---



## Code Repository
[MetaDrive](https://github.com/metadriverse/metadrive)