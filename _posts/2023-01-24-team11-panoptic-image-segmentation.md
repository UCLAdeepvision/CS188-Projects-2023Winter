---
layout: post
comments: true
title: Analysis of Panoptic Image Segmentation Performance   
author: Andrew Fantino and Nicholas Oosthuizen
date: 2023-02-26
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.

<!--more-->
{: class="table-of-content"}

* TOC
{:toc}

## Introduction

In 2019, a team from Facebook AI Research ([FAIR](https://ai.facebook.com/)) published a paper that defined a new field of computer vision called **Panoptic Image Segmentation** that combines detections of *stuff* and *things* [[1](#ref1)]. However, before we can understand what panoptic segmentation is, we must understand some background.

![Cat]({{ '/assets/images/team-11/cat.png' | relative_url }}){: style="width: 400px; max-width: 100%;"}
*Fig 1. Example cat image* [[2](#ref2)].

### Stuff & Things

***Stuff***
:   amorphous and uncountable regions of similar texture or material such as grass, sky, road defined by simply assigning a class label to each pixel in an image  [[1](#ref1)]

***Things***
:   Items in an image that could possess more than 1 countable instance defined by detecting each object and delineating it with a bounding box or segmentation mask [[1](#ref1)]

Although identifying stuff and things sound like similar problems, the deep learning models that perform the task vary substansially in datasets, details, and evaluation metrics[[1](#ref1)]

### Semantic Segmentation

Semantic segmentation is a task that indentifies stuff. **Description of how it works**

![Semantic Cat]({{ '/assets/images/team-11/sem_cat.png' | relative_url }}){: style="width: 400px; max-width: 100%;"}
*Fig 2. Semantic Segmentation on the cat image* [[3](#ref3)].

### Instance Segmentation

**Description of how it works**

![Instance Cat]({{ '/assets/images/team-11/inst_cat.png' | relative_url }}){: style="width: 400px; max-width: 100%;"}
*Fig 3. Instance Segmentation on the cat image* [[3](#ref3)].

### Panoptic Segmentation

The paper sets the groundwork for the panoptic image segmenation problem to reconcile the dichotomy between *stuff* and *things* by combining semantic and instance segmentation

![Panoptic Cat]({{ '/assets/images/team-11/pan_cat.png' | relative_url }}){: style="width: 400px; max-width: 100%;"}
*Fig 4. Panoptic Segmentation on the cat image* [[3](#ref3)].

Kirillov et al. defines a $$PS$$ *(Panoptic Score)* and the requirements for a model to be considered a "panoptic segmentation model." This task metric was designed because the standard metrics used for instance segmentation and semantic segmentation are best suited for stuff *or* things,
respectively, but not both [[1](#ref1)].

$$
PS= \frac{ \sum_{ (p,g) \in TP} IoU(p,g) }{ |TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN| }
$$

$$PS$$ is composed of the count of three sets: true positives ($$TP$$), false positives ($$FP$$), and false negatives ($$FP$$), representing matched pairs of segments, unmatched predicted segments, and unmatched ground truth segments, respectively [[1](#ref1)].

Intuitively, $$PS$$ is the average intersection over union ($$IoU$$) of matched segments divided by a penalty for segments without matches: [[1](#ref1)]

$$\frac{1}{2}|FP| + \frac{1}{2}|FN|$$

Kirillov et al. defines the panoptic segmentation format algorithm to map each pixel to an semantic class and an optional instance class. **Continue explanation**

$$
IoU(p_i,g)=\frac{|p_i\cap g|}{|p_i\cup g|}
$$

We will be evaluating and comparing multiple panoptic segmentation models on the COCO2017 dataset using MMDetection and attempt to use the results to build a new model with equal or greater Panoptic Quality.  **continue with what we are going to do** (COCO-2017 dataset)

## MMDetection Setup

We will be evaluating and modifying the panoptic segmentation models from the [MMDetection ModelZoo](https://github.com/open-mmlab/mmdetection#overview-of-benchmark-and-model-zoo) using Google Colab for development. Therefore, we will need to install MMdetection and download the COCO-2017 dataset on our Google Drive for persistent storage of the dataset.

### Install MMDetection in Google Colab

Luckily, a student group from the Winter2022 quarter of CS188 did most of the hard work with setting up MMDetection for Google Colab. They downloaded MMdetection and the COCO-2017 dataset to their Google Drive. [MMDetecton Project](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2022/02/20/team09-MMDetection.html#2-mmdetection-setup-and-data-preparation)

Unfortunately, we can't just copy-and-paste the entire setup since there have been changes to the installation steps and versions in Pytorch, mmcv, and Pillow. In addition, we need to download the additional panoptic segmentation labels.

To begin, we create a new Colab notebook that we will (hopfully) run only once to install MMDetection and download the COCO-2017 dataset. In our case, we named it `setup_mmdet_and_download_COCO.ipynb`.

First, mount your drive, and install mmcv in our colab environment. There is no need to install a special version of Pytorch or Pillow in this version of Colab.

```python
import os
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# install mmcv-full
!pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
```

Make a directory in the drive called `MMDet1` and clone the MMDetection github repository into it and install with pip.

```python
!mkdir -pv /content/drive/MyDrive/MMDet1
%cd /content/drive/MyDrive/MMaDet1

# Install mmdetection
!rm -rf mmdetection
!git clone https://github.com/open-mmlab/mmdetection.git
%cd mmdetection

!pip install -e .    
```

MMdetection is now downloaded on your Google Drive and installed in your Colab session. You should **NEVER** have to reclone the github repo in **ANY** colab notebook!

### COCO-2017 Download

Now we will download the COCO-2017 dataset with the annotations for panoptic segmentation. First, make sure you are in the mmdetection folder in your drive, then use the `download_dataset.py` script to download the base COCO-2017 dataset. Then you will need to download the panoptic annotations. Next, unzip all the newly downloaded files into the `data/coco` directory.

The following code block will take about 5 hours to run fully. If you do not wish to wait all that time and would rather unzip in bursts, you can comment out all but the unzip command that your want to execute and run the cell for each unzip command. This allows you to unzip the files in chunks of time instead of all at once.

You will know that all files have been correctly extracted if when you run the code block again there are no outputs start with `extracting`

> **Note:** <br>
    1. You will need a GPU runtime to run the `download_dataset.py` python script. <br>
    2. Sometimes the dataset will not fully unzip and not let you know. Make sure to rerun this block until there are not outputs of newly extracted filenames.

```python
# suppose data/coco/ does not exist
!mkdir -pv data/coco/

# download the coco2017 dataset
!python3 tools/misc/download_dataset.py --dataset-name coco2017 

# Adjust the dataset to include panoptic annotations
!wget -P data/coco/ http://images.cocodataset.org/annotations/image_info_test2017.zip
!wget -P data/coco/ http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip

# unzip them
!unzip -u "data/coco/annotations_trainval2017.zip" -d "data/coco/"
!unzip -u "data/coco/test2017.zip" -d "data/coco/"
!unzip -u "data/coco/train2017.zip" -d "data/coco/"
!unzip -u "data/coco/val2017.zip" -d "data/coco/"

!unzip -u data/coco/image_info_test2017.zip -d data/coco/
!unzip -u data/coco/panoptic_annotations_trainval2017.zip -d data/coco/
!unzip -u data/coco/annotations/panoptic_train2017.zip -d data/coco/annotations
!unzip -u data/coco/annotations/panoptic_val2017.zip -d data/coco/annotations
```

Finally, convert the standard COCO annotations to the panoptic annotations with the `gen_coco_panoptic_test_info.py` script.

```python
!python tools/misc/gen_coco_panoptic_test_info.py data/coco/annotations
```

You are now ready to get started with working with MMDetection in Colab. Please make sure that you install `mmcv`, `cocodataset/panopticapi`, enter the mmdetection directory and run `pip install -e .` with each new document.

## PanopticFPN with MMDetection

### Background

The Panoptic FPN was designed as a single-network baseline for the panoptic segmentation task. They do this by starting from Mask R-CNN, a popular instance segmentation model, with a Feature Pyramid Network (FPN) backbone based on ResNet. They create a minimal semantic segmentation branch using the same features of the FPN to generate a dense-pixel output [[3](#ref3)]. The author's goal is to maintain top of the line performance for segementation quality ($$SQ$$) and recognition quality ($$RQ$$) [[3](#ref3)].

![FPN]({{ '/assets/images/team-11/fpn.png' | relative_url }}){: style="width: 400px; max-width: 100%;"}
*Fig 5. Instance Segmentation on the cat image* [[1](#ref1)].

#### Feature Pyramid Network

The FPN consists of a botton up pathway and a top-down pathway. The bottom-up pathway consists of feature maps of several scales with a scaling step of 2. Each step corresponds to a residual block stage from Resnet $$\{C2, C3, C4, C5\}$$. The output of each step is the output of the activation function of the residual block (except for $$C1$$ since it is so large). The stages have strides $$\{4, 8, 16, 32\}$$ in order to downsample the feature map.

![Top_Down]({{ '/assets/images/team-11/top_down_fpn.png' | relative_url }}){: style="width: 400px; max-width: 100%;"}
*Fig 6. Instance Segmentation on the cat image* [[4](#ref4)].

The top-down pathway starts from the deepest layer of the network and progressively upsamples it while adding in transformed versions of higher-resolution features from the bottom-up pathway. The higher stages of the top-down pathway are at a smaller resolution, but semantically stronger. The purpose of the top-down pathway is to use this information to make a spatially fine and semantically stronger feature map of the input. Finally, the output of each stage of the top-down pathway is the final output of the FPN (labeled predict in Fig. 6).

#### Instance Segmentation Branch

Mask R-CNN is an extension on Faster R-CNN that adds an masking head branch to predict an binary mask for each bounding box prediction. Panoptic FPN uses the Mask R-CNN with the ResNet FPN as a backbone since it has been used as a foundation for all top entries in recent recognition challenges [[3](#ref3)].

#### Semantic Segmentation Branch

The semantic segmentation branch also builds on the FPN in parallel with the instance segmentation branch. This semantic segmentation branch was designed to be as simple as possible and so it only upsamples each output of the FPN layers to 1/4th total size, add each together, and perform a 1x1 conv with a 4x bilinear upsampling. Each upsampling layer consists of a 3x3 convolution, group norm, ReLU, and 2x bilinear upsampling. It is important to note that in addition to each of the stuff class of the dataset, the branch can also output a 'other' class for pixels that do not belong to any classes. This avoids the branch predicting the pixels belong to no class as a incorrect class [[1](#ref1)].

![Semantic_Diagram]({{ '/assets/images/team-11/semantic_diagram.png' | relative_url }}){: style="width: 400px; max-width: 100%;"}
*Fig 7. Instance Segmentation on the cat image* [[1](#ref1)].

### Setup

Before we do anything, let's make sure we have our Colab environment set up correctly. Mount you Google Drive, install `mmcv`, `cocodataset/panopticapi`, and install the mmdetection library in your Colab session. Again, it is important to note that we do not re-clone the `mmDetection` github.

```python
import os
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Install mmcv-full
!pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

# Install cocodataset/panopticapi
!pip install git+https://github.com/cocodataset/panopticapi.git

# Install mmdetection in colab environment
%cd /content/drive/MyDrive/MMDet1/mmdetection
!pip install -e .
```

### Run Test Script

MMDetection pretrained PanopticFPN Evaluation Results:
[Implementation Link](https://colab.research.google.com/drive/11MitSydv7qZ_xQkcLO4X2azTuGORjrQf#scrollTo=L-9pCPGHIkdo&uniqifier=2)

```python
**Put a code block with the test script here**
```

| Panoptic 1x ResNet50 Coco | PQ     | SQ     | RQ     | categories |
| :--------------- | :---: | :---: | :---: | :---: |
| All    | 40.248 | 77.785 | 49.312 | 133        |
| Things | 47.752 | 80.925 | 57.475 | 80         |
| Stuff  | 28.922 | 73.046 | 36.991 | 53         |

| Panoptic 3x ResNet50 Coco | PQ     | SQ     | RQ     | categories |
| :--------------- | :---: | :---: | :---: | :---: |
| All    | 42.457 | 78.118 | 51.705 | 133        |
| Things | 50.283 | 81.478 | 60.285 | 80         |
| Stuff  | 30.645 | 73.046 | 38.755 | 53         |

| ![Demo_Image]({{ '/assets/images/team-11/demo.png' | relative_url }}){: style="width: 400px; max-width: 100%;"} |

| ![Panopticfpn_1x_demo_image]({{ '/assets/images/team-11/panfpn_1_demo.png' | relative_url }}){: style="width: 400px; max-width: 100%;"} |  ![Panopticfpn_3x_demo_image]({{ '/assets/images/team-11/panfpn_3_demo.png' | relative_url }}){: style="width: 400px; max-width: 100%;"} |

*Fig 8. Example image and output from PanopticFPN. Top: test image, Left: output of PanopticFPN_1x, Right: output of PanopticFPN_3x* [[1](#ref1)].

## Maskformer with MMDetection

### Background

Semantic segmentation is often approached as per pixel classification, while instance segmentation is approached as mask classification. The key insight of Cheng, Schwing, and Kirillov to create the MaskFormer model is that "mask classification is sufficiently general to solve both semantic-level and instance-level segmentation tasks in a unified manner using the exact same model, loss, and training procedure." [[5](#ref5)] Mask classification can be used to solve semantic, instance, and panoptic segmentation together.

MaskFormer is a mask classification model which predicts a set of binary masks, each associated with one global class prediction [[5](#ref5)]. MaskFormer converts a per-pixel classification model into a mask classification model [[5](#ref5)].

#### Maskformer Architecture

![Maskformer Architecture]({{'/assets/images/team-11/maskformer_architecture.png'|relative_url}}){: style="width: 400px; max-width: 100%;"}
*Fig 9. Maskformer Architecture* [[5](#ref5)]

Maskformer breaks down into three modules: a pixel level module, a transformer module, and a segementation module. The pixel level module extracts per-pixel embeddings to generate the binary mask predictions, the transformer module computes per-segment embeddings, and the segmentation module generate the prediction pairs [5].

The pixel-level module begins with a backbone to extract features from the input image. The features are then upsampled by a pixel decoder into per-pixel embeddings. This module can be changed for any per-pixel classification-based segmentation model [[5](#ref5)].

The transfomer module uses a a Transformer decoder to compute the per-segment embeddings from the extracted image features and learnable positional embeddings [[5](#ref5)]. MaskFormer uses the standard Transfomer Decoder [[7](#ref7)]

![Transformer Architecture]({{'/assets/images/team-11/transformer_architecture.png'|relative_url}})
*Fig 10. Transformer Decoder Architecture*

A Transformer is made up of an encoder (left) and a decoder stack (right). The encoder is made up of a stack of identical layers. The layers each begin with a multihead attention layer, followed by a fully connected feed forward layer. Each sublayer has a residual connection, which is then Layer Normalized. The decoder stack is similar to the encoder stack, but an additional multihead attention layer acts on the output of the encoder stack [[7](#ref7)]. Finally a a linear layer and a softmax activation turn the attentions into output probabilities [[7](#ref7)].

Attention is computed as:

$$
Attention(Q,K,V)=softmax\left( \frac{ QK^T }{ \sqrt{d_k} } \right)V
$$

Q, K, and V are all vectors, where Q is a query, K is made up of keys, and V is made up of keys. The output is a weighted sum of the values, where the weight is computed by a compatability function between the query and the key [[7](#ref7)]. In this case, the compatability function is softmax. This means that the most likely query-key combination's value will have the highest weight, and thus the highest attention value. *d<sub>k</sub>* is the dimension of the keys and queries, and is used to scale the scale the QK product [[5](#ref5)].

Multi-head attention takes the attention function on multiple different linearly projected versions of Q, K, and V in parallel [[5](#ref5)]. These values are then cocatenated and projected again to get the final attention values [[5](#ref5)].

The segmentation module uses a linear classifier followed by softmax on the per-segment embeddings to get the class probabilities of each segment. A 2 hidden layer Multi-Layer Perceptron gets the mask embeddings from the per-segment embeddings. To get the final binary mask predictions, the model takes a dot product between the ith mask embeddings and the per-pixel embeddings from the pixel-level module. This dot product is followed by a sigmoid activation to produce the output [[5](#ref5)].

### Setup

MaskFormer follows the same setup steps as Panoptic FPN to set up MMDetection.

### Run Test Script

MMDetection pretrained MaskFormer Evaluation Results:
[Implementation Link](https://colab.research.google.com/drive/1UEj1DHPcbcxhIFO2ukt9QWSG-S1zVm5z?usp=sharing)

| MaskFormer Resnet | PQ     | SQ     | RQ     | categories |
| :--------------- | :---: | :---: | :---: | :---: |
| All    | 46.854 | 80.617 | 57.085 | 133        |
| Things | 51.089 | 81.511 | 61.853 | 80         |
| Stuff  | 40.463 | 79.269 | 49.888 | 53         |

![MaskFormer Demo Image]({{ '/assets/images/team-11/maskformer_demo.png' | relative_url }}){: style="width: 400px; max-width: 100%;"}
*Fig 12. MaskFormer Sample Output*

## Mask2Former with MMDetection

### Mask2Former Background

While MaskFormer is a universal architecture for semantic, instance, and panoptic segmentation, it is expensive to train and does not outperform specialized segmentation models [[6](#ref6)]. Masked-attention Mask Transformer, or Mask2Former, is a universal segmentation method that outperforms specialized architectures, and is simpler to train on each task. Mask2Former is similar to MaskFormer, but with several improvements. The first is using masked attention in the Transformer decoder rather than cross-attention [[6](#ref6)]. This restricts the attention on localized features centered around the predicted segments [[6](#ref6)]. Second is using "multi-scale high-resolution features", as well as optimization improvements and calculating mask loss on randomly sampled points to save training memory [[6](#ref6)].

#### Mask2Former Architecture

![Mask2Former Architecture]({{ '/assets/images/team-11/mask2former_architecture.png' | relative_url }}){: style="width: 400px; max-width: 100%;"}
*Fig 13. Mask2Former Architecture* [6]

Mask2Former follows the overall meta architecture from MaskFormer: a backbone to extract image features, a pixel decoder to upsample features into per-pixel embeddings, and a Transformer decoder to compute segments from the image features, but with the changes mentioned above [6].

##### Masked Attention

The first of these changes is using masked attention rather than cross attention in the Transformer decoder [6]. Masked attention is "a variant of cross-attention that only attends within the foreground region of the predicted mask for each query." [6] This is achieved by adding an "attention mask" to standard cross attention. [6] Cross attention computes:

$$
X_l=softmax(Q_lK_l^T)V_l+X_{l-1}
$$

*Fig 14. Cross Attention Formula* [6]

Masked attention modifies regular attention by adding an attention mask to the formula.  

$$
X_l=softmax(M_{l-1}+Q_lK_l^T)V_l+X_{l-1}
$$

*Fig 15. Mask Attention Formula* [[6](#ref6)]

$$
\begin{equation}
M_{l-1}(x,y) = \biggl\{
    \begin{array}{lr}
    0, & \text{if } M_{l-1}(x,y)=1 \\
    -\inf, & \text{otherwise}
    \end{array}
 \end{equation}
$$

*Fig 16. Attention Mask at Feature (x,y)* [[6](#ref6)]

$$X_{l-1}$$ represents a residual connection.

$$M_{l-1}$$ is the mask prediction of the previous layer, converted to binary data with threshold 0.5. This is also resized to the same dimension as $$K_l$$.

##### High-resolution Features

Higher-resolution features boost model performance, especially for small objects, but also increase computation cost [[6](#ref6)]. To gain the benefit of higher resolution images while also limiting computation cost increases, Mask2Former uses a feature pyramid of both high and low resolution features produced by the pixel decoder with resolutions 1/32, 1/16, and 1/8 of the original image [[6](#ref6)]. Each of these different feature resolutions are fed to one layer of the Transformer decoder, as can be seen in the Mask2Former architecture image. This is repeated L times in the decoder, for a total of 3L layers in the Transformer decoder [[6](#ref6)].

##### Optimization Improvements

Mask2Former makes three changes to the standard Transformer decoder design [[6](#ref6)]. The first is that Mask2Former changes the order of the self-attention and cross-attention layers (mask-attention layer), starting with the mask-attention layer rather than the self-attention layer [[6](#ref6)]. Next the query features ($$X_0$$) are made learnable, which are supervised before being used to compute masks ($$M_0$$) [[6](#ref6)]. Lastly dropout is removed, as it was not found to help performance [[6](#ref6)].

Finally to improve training efficienty, Mask2Former calculates the mask loss with samples of points rather than the whole image [[6](#ref6)]. In matching loss, a set of K points is uniformly sampled for all the ground truth and prediction masks [[6](#ref6)]. In the final loss, different sets of K points are sampled with importance sampling for each pair of predictions and ground truth [[6](#ref6)]. This sampled loss calculation reduces required memory during training by 3x [[6](#ref6)].

### Setup

Mask2Former follows the same setup steps as Panoptic FPN to set up MMDetection.

### Run Test Script

MMDetection pretrained Mask2Former Evaluation Results:
[Implementation Link](https://colab.research.google.com/drive/1P5NI9k6qnYz0G9tsCxZ446dKjvnkWVX4?usp=sharing)

| Mask2Former Resnet50 | PQ     | SQ     | RQ     | categories |
| :--------------- | :---: | :---: | :---: | :---: |
| All    | 51.865 | 83.071 | 61.591 | 133        |
| Things | 57.737 | 84.043 | 68.129 | 80         |
| Stuff  | 43.003 | 81.604 | 51.722 | 53         |

![Mask2Former Demo Image]({{ '/assets/images/team-11/mask2former_demo.png' | relative_url }}){: style="width: 400px; max-width: 100%;"}
*Fig 17. Mask2Former Sample Output*

## Evaluation

|![Plane Raw Image]({{ '/assets/images/team-11/plane.png' | relative_url}}){: style="width: 400px; max-width: 100%;"} | ![Bruinwalk Raw Image]({{ '/assets/images/team-11/bruinwalk.png'| relative_url }}){: style="width: 400px; max-width: 100%;"}|
*Fig 18. Plane Image and BruinWalk Image*

|![Plane Panoptic FPN]({{ '/assets/images/team-11/plane_pfpn.png' | relative_url}}){: style="width: 300px; max-width: 100%;"} | ![Plane MaskFormer]({{ '/assets/images/team-11/plane_maskformer.png' | relative_url}}){: style="width: 300px; max-width: 100%;"} | ![Plane Mask2Former]({{ '/assets/images/team-11/m2frmr_plane.png' | relative_url}}){: style="width: 300px; max-width: 100%;"}|
*Fig 19. (Left to Right) Panoptic FPN, MaskFormer, Mask2Former Results on Plane Image*

| ![Bruinwalk Panoptic FPN]({{ '/assets/images/team-11/bruinwalk_pfpn.png' | relative_url}}){: style="width: 400px; max-width: 100%;"} | ![Bruinwalk MaskFormer]({{ '/assets/images/team-11/bruinwalk_maskformer.png' | relative_url}}){: style="width: 400px; max-width: 100%;"} | ![Bruinwalk MaskFormer]({{ '/assets/images/team-11/bruinwalk_m2frmr.png' | relative_url}}){: style="width: 400px; max-width: 100%;"}|
*Fig 20. Panoptic FPN, MaskFormer and Mask2Former Result on Bruinwalk Image*

|![Plane Diff Panoptic FPN and MaskFormer]({{ '/assets/images/team-11/plane_diff_p_m.png' | relative_url}}){: style="width: 400px; max-width: 100%;"} | ![Bruinwalk Diff Panoptic FPN and MaskFormer]({{ '/assets/images/team-11/bruinwalk_diff_p_m.png' | relative_url}}){: style="width: 400px; max-width: 100%;"}|
*Fig 21. Panoptic FPN and MaskFormer Output Difference*

|![Plane Diff MaskFormer and Mask2Former]({{ '/assets/images/team-11/plane_diff_m_m2.png' | relative_url}}){: style="width: 400px; max-width: 100%;"} | ![Bruinwalk Diff MaskFormer and Mask2Former]({{ '/assets/images/team-11/bruinwalk_diff_pfpn_m2.png' | relative_url}}){: style="width: 400px; max-width: 100%;"}|
*Fig 22. Panoptic FPN and Mask2Former Output Differnce*

![Plane Diff MaskFormer and Mask2Former]({{ '/assets/images/team-11/plane_diff_m_m2.png' | relative_url}}){: style="width: 400px; max-width: 100%;"}
*Fig 23. MaskFormer and Mask2Former Output Difference*

## Visualizing Activations

|![Janns MaskFormer Attention]({{ '/assets/images/team-11/janns_attention_maskformer.png' | relative_url}}){: style="width: 400px; max-width: 100%;"} | ![Janns Mask2Former Attention]({{ '/assets/images/team-11/janns_attention_mask2former.png' | relative_url}}){: style="width: 400px; max-width: 100%;"}|
|![Cat MaskFormer Attention]({{ '/assets/images/team-11/cat_attention_maskformer.png' | relative_url}}){: style="width: 400px; max-width: 100%;"} | ![Cat Mask2Former Attention]({{ '/assets/images/team-11/cat_attention_mask2former.png' | relative_url}}){: style="width: 400px; max-width: 100%;"}|
|![Plane MaskFormer Attention]({{ '/assets/images/team-11/plane_attention_maskformer.png' | relative_url}}){: style="width: 400px; max-width: 100%;"} | ![Plane Mask2Former Attention]({{ '/assets/images/team-11/plane_attention_mask2former.png' | relative_url}}){: style="width: 400px; max-width: 100%;"}|
*Fig 24. MaskFormer and Mask2Former Attention*

## Summary & Conclusion

Here is where we will put a conclusion.

## References

<a name="ref1"></a>
[1] [Panoptic Segmentation, Kirillov et al.(2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.html)

<a name="ref2"></a>
[2] [What is Panoptic Segmentation and why you should care.](https://medium.com/@danielmechea/what-is-panoptic-segmentation-and-why-you-should-care-7f6c953d2a6a)

<a name="ref3"></a>
[3] [Panoptic Feature Pyramid Networks, Kirillov et al. 2019](https://arxiv.org/pdf/1901.02446.pdf)

<a name="ref4"></a>
[4] [FPN Paper](https://arxiv.org/pdf/1612.03144.pdf)

<a name="ref5"></a>
[5] [MaskFormer Paper](https://arxiv.org/pdf/2107.06278.pdf)

<a name="ref6"></a>
[6] [Mask2Former Paper](https://arxiv.org/pdf/2112.01527.pdf)

<a name="ref7"></a>
[7] [Transformer Decoder paper](https://arxiv.org/pdf/1706.03762.pdf)
