---
layout: post
comments: true
title: Panoptic Scene Graph Generation and Panoptic Segmentation
author: Alex haddad
date: 2023-01-31
---

> In this post, we'll explore what panoptic scene graph generation and panoptic segmentation are, their implementation, and potential applications.

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## Project Overview

Panoptic scene graph generation (PSG) aims to generate a representation of the semantic relationships between objects in an image. Panoptic segmentation aims to divide an image into objects or separate parts using pixel-level separation; it is used in PSG.

## Panoptic Scene Graph Generation

Panoptic scene graph generation predicts a set of triplets, where each triplet has a subject, relationship, and object.

## Panoptic Segmentation & Scene Graph Example Visualization

First, we need to set up our environment. Clone the OpenPSG repository and install the following dependencies:

```sh
git clone https://github.com/Jingkang50/OpenPSG.git
pip install mmcv-full==1.4.3 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
pip install openmim
mim install mmdet==2.20.0
pip install git+https://github.com/cocodataset/panopticapi.git
pip install pycocotools
pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
```

We'll also need to download the data from this [link](https://entuedu-my.sharepoint.com/personal/jingkang001_e_ntu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjingkang001%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2Fopenpsg%2Fdata&ga=1).

Now we need to import the necessary libraries:

```python
%load_ext autoreload
%autoreload 2

import json
import sys
import matplotlib.pyplot as plt
from panopticapi.utils import rgb2id
from PIL import Image

# add OpenPSG to the path so we can import it
sys.path.append('OpenPSG')
from openpsg.utils.vis_tools.detectron_viz import Visualizer
```

Next, we'll initialize the dataset.

```python
with open("openpsg-data/psg/psg.json") as f:
	psg_dataset_file = json.load(f)

print("keys:", list(psg_dataset_file.keys()))

thing_classes = psg_dataset_file["thing_classes"]
stuff_classes = psg_dataset_file["stuff_classes"]
psg_object_classes = thing_classes + stuff_classes
psg_rel_classes = psg_dataset_file["predicate_classes"]
psg_dataset = {d["coco_image_id"]: d for d in psg_dataset_file["data"]}
```

I chose image 71 from the dataset. We can use the following code to visualize the original image.

```python
image_id = "71"
image_data = psg_dataset[image_id]

image_path = f"openpsg-data/coco/{image_data['file_name']}"
image = Image.open(image_path).convert("RGB")
plt.imshow(image)
plt.show()
```

![image](/assets/images/team51/original-image.png)

To visualize the panoptic segmentation, we can use the following code:

```python
seg_map_path = f"openpsg-data/coco/{image_data['pan_seg_file_name']}"
seg_map = Image.open(seg_map_path).convert("RGB")
plt.imshow(seg_map)
plt.show()

seg_map = rgb2id(seg_map)
```

![image](/assets/images/team51/segment-map.png)

We can overlay the segmentation map and the original image with this code:

```python
viz = Visualizer(image)
viz.overlay_instances(
    labels=labels_coco,
    masks=masks,
)
image_viz = viz.get_output().get_image()
plt.figure(figsize=(10,10))
plt.imshow(image_viz)
plt.axis('off')
plt.show()
```

![image](/assets/images/team51/original-image-map-overlay.png)

Lastly, the following code will print the scene graph:

```python
for s_idx, o_idx, rel_id in image_data["relations"]:
    s_label = labels_coco[s_idx]
    o_label = labels_coco[o_idx]
    rel_label = psg_rel_classes[rel_id]
    print(s_label, rel_label, o_label)
```

In this example, we get

```txt
car driving on road
train on railroad
train driving on railroad
house beside road
road beside house
sky-other-merged over building-other-merged
```

## Three Most Relevant Research Papers

1. [Panoptic Scene Graph Generation](https://arxiv.org/abs/2207.11247) ([Code](https://github.com/Jingkang50/OpenPSG))
2. [Panoptic Segmentation](https://arxiv.org/abs/1801.00868) ([Code](https://github.com/cocodataset/panopticapi))
3. [SOGNet: Scene Overlap Graph Network for Panoptic Segmentation](https://arxiv.org/abs/1911.07527) ([Code](https://github.com/LaoYang1994/SOGNet))
