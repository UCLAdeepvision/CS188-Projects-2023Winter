---
layout: post
comments: true
title: Object Boxes
author: Ryan Vuong and Travis Graening
date: 2022-01-29
---


> In this project, we wish to see how the accuracy of a machine learning model to recognize images containing certain objects can be improved. We plan to do this by highlighting smaller portions of the given images to narrow our model's focus onto more relevant parts of the image. This way, extraneous unhelpful noise will be dramatically reduced


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## References and Code Bases
Jimmy Ba, Volodymyr Mnih, Koray Kavukcuoglu: "Multiple Object Recognition with Visual Attention"
-article: https://paperswithcode.com/paper/multiple-object-recognition-with-visual
-code: https://paperswithcode.com/paper/multiple-object-recognition-with-visual#code

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