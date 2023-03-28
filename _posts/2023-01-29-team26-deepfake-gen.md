---
layout: post
comments: true
title: Deepfake Generation
author: Freddy Aguilar, Luis Frias
date: 2022-01-30
---

> This article investigates the topic of DeepFake Generation, comparing two models, GAN and CNN, and analyzing their similarities and differences in the context of Faceswap. The study was conducted by implementing the Deepfacelab model and comparing it with the Faceswap CNN model. The hypothesis is that GAN will perform better due to its traditional generative model nature that specializes in image generation, whereas CNNs are designed for image processing tasks, such as object recognition and segmentation. The results of the study shed light on the effectiveness of these models for DeepFake Generation.

> Deepfake Generation
<!--more-->
{: class="table-of-content"}
* TOC
{:toc}


# Introduction
Recent years have seen a tremendous advancement in deep fake technology, with new models and algorithms emerging that enable more convincing and realistic picture swaps. There are many uses for the skill of convincingly and flawlessly switching someone's visage, from entertainment to political influence. But as deep fake technology spreads, it's critical to comprehend both its advantages and disadvantages. The Generative Adversarial Networks (GANs) and a modified version of the VGG16 CNN architecture are two deep fake generation techniques for picture swaps that are compared and contrasted in this article. The essay will assess each model's performance and go through its merits and cons. We implemented the deepfacelab model and used output data from the CNN model. It's vital to highlight that we concentrated our efforts on the deepfacelab model due to  limitations of implementing models. Google has made it difficult AI systems that can be used to generate deepfakes on its Google Colaboratory platform due to ethical reasons. Nonetheless, we proceeded with the comparsions. We expect that the GAN model will produce higher quality images due to its capability to generate high-quality images with complex textures and details, which is crucial for creating convincing deepfakes. This study aims to provide insights into the current state of deep fake creation for picture swaps and contribute to ongoing discussions regarding its appropriate use by comparing these models side by side.


# Deep Fake Generation?
Deep fake generation is the process of producing synthetic media that closely resembles the appearance and behavior of actual people is known as "deep fake generation. This technology can produce incredibly lifelike images, films, and audio. For the intention of limiting the scope of the article, we will focus on pictures swaps. Face swapping is a specific type of deep fake generation that involves replacing one person's face with another person's face in an image or video. 

Face swapping typically entails feeding the network pairs of photos, each of which has the face of one person and the face of another. The network then learns to recognize the features and characteristics of each person's face and the relationship between them.

Once trained, the face swap model can be used to create new pictures or movies with the faces of two persons switched. It is feasible to create highly realistic face swaps that are challenging to discern from actual photographs by carefully tweaking the network's parameters after training it on a sizable collection of facial images.

 ![Overview of Training](/assets/images/team26/Face_swap_ex_left.jpg)
  ![Overview of Training](/assets/images/team26/Face_swap_ex_Right.jpg)
# Deep Face Lab - GAN Model

The piepline has three main components: Extraction, Training, Conversion

 - Extraction is the first phase in DFL, which contains many algorithms and processing parts, i.e. face detection, face alignment, and face segmentation. After the procedure of Extraction, user will get the aligned faces with precise mask and facial landmarks 

 ![Overview of extraction](/assets/images/team26/Overview_extraction.png)

- The training process involves an Encoder and Inter, which have shared weights between the source and destination, along with a separate Decoder for each. This allows for the generalization of the source and destination through the shared Encoder and Inter, effectively solving the unpaired problem. The Inter extracts the latent codes for both the source and destination, denoted as Fsrc and Fdst, respectively.
DFL employs a mixed loss, combining DSSIM (structural dissimilarity) [18] and MSE, to optimize its performance. This combination of losses allows for the benefits of both methods: DSSIM improves generalization for human faces, while MSE enhances clarity. Ultimately, this mixed loss strikes a balance between generalization and clarity.

 ![Overview of Training](/assets/images/team26/Overview_Training.png)

 - The Conversion step allows users to swap faces between the source and destination photos. This is accomplished by transforming the created face and its mask from the destination Decoder to the target image's original place in the source image. The target image is then perfectly blended with the realigned face using a variety of color transfer methods from DFL. The target image and the re-aligned face are combined to create the final blended image.

 ![Overview of Conversion](/assets/images/team26/Overview_Conversion.png)

 # Facewap - VGG16 CNN Model
- The FaceSwap uses the CNN VGG16 model to identify important characteristics of facial features. This process is accompanied by a series of alignment and realignment steps, which are crucial in ensuring the accuracy of the feature extraction. These alignment and realignment steps are an integral part of the overall process of identifying facial features. These faces would then be saved to be used in the next step.
  ![VGG16 Face Dectction](/assets/images/team26/VGG16.png)


- The architecture of this network involves multiple branches that operate on different versions of the input image, with each branch containing blocks of zero-padded convolutions and linear rectification. The branches are combined through nearest-neighbor upsampling and concatenation along the channel axis. The last branch has a 1x1 convolution and outputs 3 color channels.

- This network is designed for 128x128 inputs and has 1 million parameters. It can be easily adapted for larger inputs, such as 256x256 or 512x512, by adding extra branches. The output of the network comes from the branch with the highest resolution.

- To train the network on larger images, it is convenient to first train it on 128x128 inputs and then use it as a starting point for larger images. However, the availability of high quality image data for model training is a limiting factor.

  ![transformation network](/assets/images/team26/transformation_network.png)


# DeepFaceLab Implementation

## Implementation Overview 
 For the deepfacelab implementation we choose google collab despite the [ban](https://www.techradar.com/news/google-is-cracking-down-hard-on-deepfakes). We believe they only allow google collab pro users to implement such models. Given only one of us had google collab pro, we were only able to implement DeepfaceLab on collab using the given guide on the repository. We made an attempt to implement deepfacelab with the [linux based repository](https://github.com/nagadit/DeepFaceLab_Linux) but we had issues training the model on a google cloud virtual machine. There were specific hardware and software requirements not available on google cloud. 

 There are 4 main steps involved in generating an output:

 - File set up: you need a workspace file with a data_src, data_dst, and model sub-folders.

 - Collecting data: to implement the deepfacelab model you must collect two quality videos that the model can train on. You need to ensure that both videos capture the same facial profiles of each person you are doing a swap on. You labels these as "data_src.mp4" and "data_dst.mp4" and store in the workspace file. The other option is to utilize facesets for source and dst.

 - Preprocessing data: The videos need to be extracted into frames, sorted, resized, and aligned.  We attempted to do this on collab but it took hours since it runs on the cpu. The runtime resets after 12 hours so we went ahead and preprocesed the data on a linux server using the available scripts to preprocess. We decided to use celebrity and non-celebrity faceset 
 - Train: The final step is to train the model. We used Sparse Auto Encoder HD. The standard model and trainer for most deepfakes.

## Implementation
There are a multiple of different steps possible in the [collab notebook](https://colab.research.google.com/github/chervonij/DFL-Colab/blob/master/DFL_Colab.ipynb#scrollTo=JNeGfiZpxlnz) given. We only utilized the following code blocks/steps.

 1. Install DeepfaceLab repository (in collab)

 2. Upload workspace to google drive with the given file structure (two videos, facesets, data_src, data_dst, and model folder) 

 3. import workspace from drive 

 
 ```
Mode = "workspace" #@param ["workspace", "data_src", "data_dst", "data_src aligned", "data_dst aligned", "models"]
Archive_name = "workspace.zip" #@param {type:"string"}

#Mount Google Drive as folder
from google.colab import drive
drive.mount('/content/drive')

def zip_and_copy(path, mode):
  unzip_cmd=" -q "+Archive_name
  
  %cd $path
  copy_cmd = "/content/drive/My\ Drive/"+Archive_name+" "+path
  !cp $copy_cmd
  !unzip $unzip_cmd    
  !rm $Archive_name

if Mode == "workspace":
  zip_and_copy("/content", "workspace")
elif Mode == "data_src":
  zip_and_copy("/content/workspace", "data_src")
elif Mode == "data_dst":
  zip_and_copy("/content/workspace", "data_dst")
elif Mode == "data_src aligned":
  zip_and_copy("/content/workspace/data_src", "aligned")
elif Mode == "data_dst aligned":
  zip_and_copy("/content/workspace/data_dst", "aligned")
elif Mode == "models":
  zip_and_copy("/content/workspace", "model")
  
print("Done!")
  ```

4. unpack faceset
```
Folder = "data_src" #@param ["data_src", "data_dst"]
Mode = "unpack" #@param ["pack", "unpack"]

cmd = "/content/DeepFaceLab/main.py util --input-dir /content/workspace/" + \
      f"{Folder}/aligned --{Mode}-faceset"

!python $cmd
```

5. training SAEHD model and backup every hour due to runtime limitations

```
#@title Training
Model = "SAEHD" #@param ["SAEHD", "AMP", "Quick96", "XSeg"]
Backup_every_hour = True #@param {type:"boolean"}
Silent_Start = True #@param {type:"boolean"}

%cd "/content"

#Mount Google Drive as folder
from google.colab import drive
drive.mount('/content/drive')

import psutil, os, time

p = psutil.Process(os.getpid())
uptime = time.time() - p.create_time()

if (Backup_every_hour):
  if not os.path.exists('workspace.zip'):
    print("Creating workspace archive ...")
    !zip -0 -r -q workspace.zip workspace
    print("Archive created!")
  else:
    print("Archive exist!")

if (Backup_every_hour):
  print("Time to end session: "+str(round((43200-uptime)/3600))+" hours")
  backup_time = str(3600)
  backup_cmd = " --execute-program -"+backup_time+" \"import os; os.system('zip -0 -r -q workspace.zip workspace/model'); os.system('cp /content/workspace.zip /content/drive/My\ Drive/'); print('Backed up!') \"" 
elif (round(39600-uptime) > 0):
  print("Time to backup: "+str(round((39600-uptime)/3600))+" hours")
  backup_time = str(round(39600-uptime))
  backup_cmd = " --execute-program "+backup_time+" \"import os; os.system('zip -0 -r -q workspace.zip workspace'); os.system('cp /content/workspace.zip /content/drive/My\ Drive/'); print('Backed up!') \"" 
else:
  print("Session expires in less than an hour.")
  backup_cmd = ""
    
cmd = "DeepFaceLab/main.py train --training-data-src-dir workspace/data_src/aligned --training-data-dst-dir workspace/data_dst/aligned --pretraining-data-dir pretrain --model-dir workspace/model --model "+Model

if Model == "Quick96":
  cmd+= " --pretrained-model-dir pretrain_Q96"

if Silent_Start:
  cmd+= " --silent-start"

if (backup_cmd != ""):
  train_cmd = (cmd+backup_cmd)
else:
  train_cmd = (cmd)

!python $train_cmd
```
6. Merge frames 
```
#@title Merge
Model = "SAEHD" #@param ["SAEHD", "AMP", "Quick96" ]

cmd = "DeepFaceLab/main.py merge --input-dir workspace/data_dst --output-dir workspace/data_dst/merged --output-mask-dir workspace/data_dst/merged_mask --aligned-dir workspace/data_dst/aligned --model-dir workspace/model --model "+Model

%cd "/content"
!python $cmd
```

7. Video results 

```
#@title Get result video 
Mode = "result video" #@param ["result video", "result_mask video"]
Copy_to_Drive = True #@param {type:"boolean"}


if Mode == "result video":
  !python DeepFaceLab/main.py videoed video-from-sequence --input-dir workspace/data_dst/merged --output-file workspace/result.mp4 --reference-file workspace/data_dst.mp4 --include-audio
  if Copy_to_Drive:
    !cp /content/workspace/result.mp4 /content/drive/My\ Drive/
elif Mode == "result_mask video":
  !python DeepFaceLab/main.py videoed video-from-sequence --input-dir workspace/data_dst/merged_mask --output-file workspace/result_mask.mp4 --reference-file workspace/data_dst.mp4
  if Copy_to_Drive:
    !cp /content/workspace/result_mask.mp4 /content/drive/My\ Drive/

```

# FaceSwap Implementation 
The folowing [colab](https://colab.research.google.com/drive/1_j_0N9uCR47ms5paXTKgQVTq1M4GX-9M?usp=sharing) will be used as the setup but there is also a [locally based](https://github.com/deepfakes/faceswap/blob/master/INSTALL.md) installation method based on your hardware configuration.

**NOTE: Colab does not support deepfake training unless you pay premium and keep the session active throughout the whole training prcoess. The local based repo provided by u/deepfake is better or run the repo off a virtual machine with a GPU.**

The process can be divided into three parts:

1. Extract
    
    This step will take photos from an `src` folder and extract faces into an `extract` folder. The `extract` folder will contains the recognized faces from the respective `src` folder. Be sure to erase any incorrectly recognized faces from the `extract` folder.

2. Train

    This step will take photos from two folders containing pictures of both faces and train a model that will be saved inside the `models` folder. The `models` folder will contain the trained model that will be used in the next step. If you have low VRAM, you can use the `lowmem` option to reduce the VRAM usage or you can lower batch size with a set cap of iterations.

3. Convert

    This step will take photos from original folder and apply new faces into `modified` folder. The `modified` folder will contain the new faces applied to the original photos. This can also work with videos but be sure to have proper CPU cooling as this maximizes CPU usage.



# Model Comparison 

Deepfakes trained on GANs typically result in results that are more visually convincing than deepfakes trained on conventional CNNs like VGG16. This is due to the fact that GANs were created primarily for creating realistic images by playing an adversarial game between a generator network and a discriminator network. The generator tries to create realistic images while the discriminator tries to differentiate between real and fake images. Over time, the generator learns to produce increasingly realistic images that can fool the discriminator.

The following image below shows the results between the two facesets using Deepfakelab. 

![results](/assets/images/team26/results_deepfacelab.png)

Traditional CNNs, such the VGG16, on the other hand, are more commonly employed for classification jobs and might not be designed to produce realistic images. Although they can still be utilized for deepfake generation, they could need more training data or longer training cycles to match GAN-based techniques' levels of visual realism.

Results from faceswap 

![results](/assets/images/team26/faceswap_results.png)

The GAN based deepfake models perform better than CNN based models. Although both of the models are not as perfect as the more popular videos, we have limited hardware. We used up our cloud educational credit and had to use a local 3070 with not enough VRAM or time to train the models better. Given more resources it is likely both deepfake models would be better. The GAN model has better results around the mouth and eyes due to its ability to create more realistic images with its generator and discriminator network. 

Gan vs Non Gan Results

![Gan vs. Non Gan](/assets/images/team26/gan_vs_non_gan.png)



# References

**First order Motion Model**

- Repository:
 First Order Motion Model. GitHub repository, https://github.com/AliaksandrSiarohin/first-order-model, 2019.

- Paper:
Siarohin, Aliaksandr, Stéphane Lathuilière, Sergey Tulyakov, Elisa Ricci, and Nicu Sebe. (2019). "First Order Motion Model for Image Animation". https://papers.nips.cc/paper/2019/hash/31c0b36aef265d9221af80872ceb62f9-Abstract.html


**Face Swap**
- Repository: DeepFakes. FaceSwap. Github repository,  https://github.com/deepfakes/faceswap, 2018.

- Paper: Korshunova, I., Shi, W., Dambre, J., & Theis, L. (2017). "Fast Face-swap Using Convolutional Neural Networks".  https://arxiv.org/abs/1611.09577.

**DeepFaceLab**
- Repository: DeepFaceLab. GitHub repository, https://github.com/iperov/DeepFaceLab, 2020.
- Paper: Perov, I., Liu, K., Umé, C., Facenheim, C. S., Jiang, J., Zhou, B., Gao, D., Chervoniy, N., Zhang, S., Marangonda, S., dpfks, M., RP, L., Wu, P., & Zhang, W. (2020). "DeepFaceLab: A simple, flexible and extensible face swapping framework". https://arxiv.org/pdf/2005.05535v4.pdf.

**Google Collab Ban**
- TechRadar. "Google Is Cracking Down Hard on Deepfakes." TechRadar, 25 Sept. 2020, https://www.techradar.com/news/google-is-cracking-down-hard-on-deepfakes.

**[Collab Notebook](https://colab.research.google.com/github/chervonij/DFL-Colab/blob/master/DFL_Colab.ipynb#scrollTo=JNeGfiZpxlnz,)**




