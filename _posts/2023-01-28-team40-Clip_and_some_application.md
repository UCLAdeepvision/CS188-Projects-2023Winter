---
layout: post
comments: true
title: Post Template
author: Ning Xu
date: 2023-01-28
---


> CLIP (Contrastive Language-Image Pre-training) is a neural network model developed by OpenAI that combines the strengths of both computer vision and natural language processing to improve image recognition and object classification.x`
CLIP has been shown to achieve state-of-the-art results on a wide range of image recognition benchmarks, and it has been used in various applications such as image captioning and image search.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Main Content
Your survey starts here. You can refer to the [source code](https://github.com/ningwebbeginner/CS188-Projects-2023Winter/tree/main/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## Reference
Please make sure to cite properly in your work, for example:

[1] Radford, Alec et al. “*Learning Transferable Visual Models From Natural Language Supervision*.” International Conference on Machine Learning 2021.

[2] Luo, Huaishao et al. “*CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval*.” Neurocomputing 508 (2021): 293-304.

[3] Zhang, Renrui et al. “*PointCLIP: Point Cloud Understanding by CLIP*.” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021): 8542-8552.

---
