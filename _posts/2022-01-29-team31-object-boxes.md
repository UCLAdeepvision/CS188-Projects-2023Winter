---
layout: post
comments: true
title: Object Boxes
author: Ryan Vuong and Travis Graening
date: 2022-01-29
---


> In this project, we wish to evaluate the accuracy of a machine learning model to recognize images containing certain objects and explore any possible improvements. We plan to do this by segmenting the given images to narrow our model's focus onto more relevant parts of the image. This way, extraneous unhelpful noise will be dramatically reduced.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## References and Code Bases

**Multiple Object Recognition with Visual Attention**

Jimmy Ba, Volodymyr Mnih, Koray Kavukcuoglu
- [Paper](https://paperswithcode.com/paper/multiple-object-recognition-with-visual)
- [Code](https://paperswithcode.com/paper/multiple-object-recognition-with-visual#code)

**Scalable Object Detection Using Neural Networks**

Dumitru Erhan, Christian Szegedy, Alexander Toshev, Dragomir Anguelov
- [Paper](https://paperswithcode.com/paper/scalable-object-detection-using-deep-neural)
- [Code](https://paperswithcode.com/paper/scalable-object-detection-using-deep-neural#code)

**Objects as Points**

Xingyi Zhou, Dequan Wang, Philipp Krähenbühl
- [Paper](https://www.semanticscholar.org/paper/Objects-as-Points-Zhou-Wang/6a2e2fd1b5bb11224daef98b3fb6d029f68a73f2)
- [Code](https://github.com/xingyizhou/CenterNet)

**Pointly-Supervised Instance Segmentation**

Bowen Cheng, Omkar Parkhi, Alexander Kirillov: Pointly-Supervised Instance Segmentation
- [Paper](https://arxiv.org/pdf/2104.06404.pdf)
- [Code](https://github.com/facebookresearch/detectron2/tree/main/projects/PointSup)

**End-to-End Object Detection with Transformers**

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko
- [Paper](https://arxiv.org/pdf/2005.12872.pdf)
- [Code](https://github.com/facebookresearch/detr#detr-end-to-end-object-detection-with-transformers)

## Basic Syntax
### Image


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

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

---