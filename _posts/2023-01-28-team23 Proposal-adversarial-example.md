---
layout: post
comments: true
title: Adversarial examples to DNNs
author: Yuxiao Lu
date: 2023-01-28
---


> **Topic:**
>
> DNN driven image recognition have been used in many real-world scenarios, such as for detection of road case or people. However, the DNNs could be vulnerable to adversarial examples (AEs), which are designed by attackers and can mislead the model to predict incorrect outputs while hardly be distinguished by human eyes. This blog aims to introduce popular  AE generation methods and reproduce their experiments to show the potential issues in the  security-critical deep-learning applications.

<!--more-->
{: class="table-of-content"}

* TOC
{:toc}

## Main Content/Code Repository
Here are some related work and their corresponding repository:

Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples (https://github.com/deepmind/deepmind-research)

Technical Report on the CleverHans v2.1.0 Adversarial Examples Library (https://github.com/tensorflow/cleverhans)

Adversarial Examples Improve Image Recognition (https://github.com/tensorflow/tpu)

EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples (https://github.com/cleverhans-lab/cleverhans/blob/master/cleverhans_v3.1.0/cleverhans/attacks/elastic_net_method.py)

Indicators of Attack Failure: Debugging and Improving Optimization of Adversarial Examples (https://github.com/Trusted-AI/adversarial-robustness-toolbox)

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

## Reference/Research Papers

[1] Gowal, S., Qin, C., Uesato, J., Mann, T., & Kohli, P. (2020). Uncovering the limits of adversarial training against norm-bounded adversarial examples. *arXiv preprint arXiv:2010.03593*. 

[2] Papernot, N., Faghri, F., Carlini, N., Goodfellow, I., Feinman, R., Kurakin, A., ... & McDaniel, P. (2016). Technical report on the cleverhans v2. 1.0 adversarial examples library. *arXiv preprint arXiv:1610.00768*. 

[3] Xie, C., Tan, M., Gong, B., Wang, J., Yuille, A. L., & Le, Q. V. (2020). Adversarial examples improve image recognition. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 819-828). 

[4] Chen, P. Y., Sharma, Y., Zhang, H., Yi, J., & Hsieh, C. J. (2018, April). Ead: elastic-net attacks to deep neural networks via adversarial examples. In *Proceedings of the AAAI conference on artificial intelligence* (Vol. 32, No. 1). 

[5] Pintor, M., Demetrio, L., Sotgiu, A., Demontis, A., Carlini, N., Biggio, B., & Roli, F. (2021). Indicators of attack failure: Debugging and improving optimization of adversarial examples. *arXiv preprint arXiv:2106.09947*. 

---

