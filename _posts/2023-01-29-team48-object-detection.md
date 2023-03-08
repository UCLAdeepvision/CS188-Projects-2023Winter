---
layout: post
comments: true
title: Object Detection
author: Jay Jay Phoemphoolsinchai
date: 2022-01-29
---

# Modern Object Detection
> Object detection is an incredibly important fundamental idea; many advanced and sophisticated tasks are directly impacted by the performance of an object detection model. In this article, we take a close look at new developments in the object detection field of computer vision that positively affect the capabilities of object detection algorithms.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}


# Goal
To see if different characteristics of object detection models can lead to significantly differing performances for different classes, and also compare visually to see if there are any noticable ways in which models differ from each other in detection.

# Models
A variety of models with different backbones and necks will be chosen. For this preview of the final demo, only one model is shown for brevity, but multiple models will be used in the final demo.

# Planned Experiments
First, images for particular classes will be downloaded from the COCO dataset. Then, each model will run on these images and performance will be compared to see which one is the best, and also if there was any significant standout (i.e. if any model was an outlier compared to others).

# Reasoning
If any conclusive results can be drawn from these experiments, it will show that there is a possibility that certain components of object detection models may lend themselves more to particular use cases than others.

# Example Code
## Retrieving images
```python
from pycocotools.coco import COCO
from collections import defaultdict
!wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
!unzip -o annotations_trainval2017.zip

ANNOTATIONS_PATH = "annotations/instances_val2017.json"
c = COCO(annotation_file=ANNOTATIONS_PATH)

cat_ids = c.getCatIds()

id_to_name = {}
for id in cat_ids:
  id_to_name[id] = c.loadCats(id)[0]['name']
  
# See which categories you want
# print(id_to_name)

ids_to_grab = [1] # Put the ID for the categories you want here
id_to_imIds = {}
for id in ids_to_grab:
  id_to_imIds[id] = c.getImgIds(catIds=id)

imId_to_imURLS = defaultdict(list)
for imId in id_to_imIds[id]:
  im_URL = c.loadImgs(imId)[0]['coco_url']
```
See the full code at **WWW.FINALDEMO.COM (PUT LINK HERE)**
## Predicting
```python
import torch
import detectron2

from detectron2 import model_zoo as mz
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# For example, let's use this model
MODEL_LINK = 'COCO-Detection/faster_rcnn_R_50_C4_1x.yaml'
cfg = get_cfg()
cfg.merge_from_file(mz.get_config_file(MODEL_LINK))
cfg.MODEL.WEIGHTS = mz.get_checkpoint_url(MODEL_LINK)
predictor = DefaultPredictor(cfg)

for image in images: # images is a list containing images read by cv2
  preds = predictor(image)
```
See the full code at **WWW.FINALDEMO.COM (PUT LINK HERE)**




# References
## Relevant Papers 
1. **InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions**
- [Paper](https://arxiv.org/abs/2211.05778v2)
- [Code](https://github.com/opengvlab/internimage)

2. **EVA: Exploring the Limits of Masked Visual Representation Learning at Scale**
- [Paper](https://arxiv.org/abs/2211.07636v2)
- [Code](https://github.com/baaivision/eva)

3. **Diffusion Models Beat GANs on Image Synthesis**
- [Paper](https://arxiv.org/abs/2211.12860v1)
- [Code](https://github.com/sense-x/co-detr)

---
