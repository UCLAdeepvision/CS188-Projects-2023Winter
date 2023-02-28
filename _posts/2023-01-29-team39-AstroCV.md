---
layout: post
comments: true
title: Galaxy Detection/Classification with Computer Vision
author: Euibin Kim
date: 2023-01-29

---

> In this blog, I will share my experience in developing a machine learning model that detects and/or classifies galaxies from public datasets from the Sloan Digital Sky Survey (SDSS) and Galaxy Zoo while taking CS 188: Deep Learning for Computer Vision at UCLA.

<!--more-->

{: class="table-of-content"}

- TOC
  {:toc}

---

## Introduction

Astronomical datasets are constantly increasing in size and complexity. The modern generation of integral field units (IFUs) are generating about 60 GB of data per night while imaging instruments are generating 300 GB of data per night. The Large Synoptic Survey Telescope (LSST, now called Vera C. Rubin Observatory) is under construction in Chile and it is expected to start full operations in 2023. With a wide 9.6 square degree field of view 3.2 Gigapixel camera, LSST will generate about 20 TB of data per night (González et al).  <br>

In this project, I used the Galaxy Zoo (Lintott et al) data, one of the most successful citizen project in Astronomy where hundreds of thousands of volunteers classified images of nearly 900,000 galaxies obtained from the SDSS survey, to finetune the ResNet18 model from the library. Then I'll be using Yolo3 based galaxy detection algorithm implemented by González, Muñoz, and Hernández to detect galaxies from the R1001ED field, the one I'm currently doing research on with Dr. Rich from the Physics & Astronomy department to detect Lyman-alpha line emitting galaxies (which requires spectroscopic analyses), and use the model I trained with the Galaxy Zoo to classify them into 6 different categories  (Elliptical, On_Edge, Spiral_Barred, Spiral, Irregular, and Star/Artifact).

## Data Exploration

https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/data

- The data contains 61578 galaxy images/labels for training and 79975 galaxy images/labels for testing

### Labels

- Each galaxy has a matching entry in a separate csv file where each column value represents the ratio of how volunteers (Galaxy Zoo) responded to questions in a following decision tree.

![DT]({{'/assets/images/team39/decision-tree.png'|relative_url}})

{: style="width: 800px; max-width: 100%;"}

<br>





| index | GalaxyID | Class1\.1 | Class1\.2 | Class1\.3 | Class2\.1    | Class2\.2    | Class3\.1    | Class3\.2    | Class4\.1    | Class4\.2    |
| ----- | -------- | --------- | --------- | --------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| 0     | 100008   | 0\.383147 | 0\.616853 | 0\.0      | 0\.0         | 0\.616853    | 0\.038452149 | 0\.578400851 | 0\.418397819 | 0\.198455181 |
| 1     | 100023   | 0\.327001 | 0\.663777 | 0\.009222 | 0\.031178269 | 0\.632598731 | 0\.467369636 | 0\.165229095 | 0\.591327989 | 0\.041270741 |
| 2     | 100053   | 0\.765717 | 0\.177352 | 0\.056931 | 0\.0         | 0\.177352    | 0\.0         | 0\.177352    | 0\.0         | 0\.177352    |
| 3     | 100078   | 0\.693377 | 0\.238564 | 0\.068059 | 0\.0         | 0\.238564    | 0\.109493481 | 0\.129070519 | 0\.189098232 | 0\.049465768 |
| 4     | 100090   | 0\.933839 | 0\.0      | 0\.066161 | 0\.0         | 0\.0         | 0\.0         | 0\.0         | 0\.0         | 0\.0         |

<br>





| Class5\.1 | Class5\.2    | Class5\.3    | Class5\.4    | Class6\.1 | Class6\.2 | Class7\.1    | Class7\.2    | Class7\.3    | Class8\.1 |
| --------- | ------------ | ------------ | ------------ | --------- | --------- | ------------ | ------------ | ------------ | --------- |
| 0\.0      | 0\.104752126 | 0\.512100874 | 0\.0         | 0\.054453 | 0\.945547 | 0\.201462524 | 0\.181684476 | 0\.0         | 0\.0      |
| 0\.0      | 0\.236781072 | 0\.160940708 | 0\.23487695  | 0\.189149 | 0\.810851 | 0\.0         | 0\.135081824 | 0\.191919176 | 0\.0      |
| 0\.0      | 0\.11778975  | 0\.05956225  | 0\.0         | 0\.0      | 1\.0      | 0\.0         | 0\.74186415  | 0\.02385285  | 0\.0      |
| 0\.0      | 0\.0         | 0\.113284024 | 0\.125279976 | 0\.320398 | 0\.679602 | 0\.408599439 | 0\.284777561 | 0\.0         | 0\.0      |
| 0\.0      | 0\.0         | 0\.0         | 0\.0         | 0\.029383 | 0\.970617 | 0\.494587282 | 0\.439251718 | 0\.0         | 0\.0      |

<br>



---

- To narrow down categories into 6 aforementioned categories, I used a mapping,
  
  - {'Class1.1': 'Elliptical', 'Class1.3': 'Star/Artifact', 'Class2.1': 'On Edge', 'Class3.1': 'Spiral_Barred', 'Class4.1': 'Spiral', 'Class8.4': Irregular}
    
    <br>
    
    
  
  | GalaxyID | Elliptical | Star/Artifact | On\_Edge     | Spiral\_Barred | Spiral       | Irregular  |
  | -------- | ---------- | ------------- | ------------ | -------------- | ------------ | ---------- |
  | 100008   | 0\.383147  | 0\.0          | 0\.0         | 0\.038452149   | 0\.418397819 | 0\.0272265 |
  | 100023   | 0\.327001  | 0\.009222     | 0\.031178269 | 0\.467369636   | 0\.591327989 | 0\.0       |
  | 100053   | 0\.765717  | 0\.056931     | 0\.0         | 0\.0           | 0\.0         | 0\.0       |
  | 100078   | 0\.693377  | 0\.068059     | 0\.0         | 0\.109493481   | 0\.189098232 | 0\.0961194 |
  | 100090   | 0\.933839  | 0\.066161     | 0\.0         | 0\.0           | 0\.0         | 0\.0       |
  | 100122   | 0\.738832  | 0\.023009     | 0\.0         | 0\.0           | 0\.0         | 0\.098965  |
  | 100123   | 0\.462492  | 0\.081475     | 0\.0         | 0\.0           | 0\.0         | 0\.0       |
  | 100128   | 0\.687783  | 0\.023873     | 0\.0         | 0\.069098179   | 0\.0         | 0\.0       |
  | 100134   | 0\.021834  | 0\.001214     | 0\.021750859 | 0\.313076726   | 0\.546490632 | 0\.4502247 |
  | 100143   | 0\.269843  | 0\.0          | 0\.730157    | 0\.0           | 0\.0         | 0\.0       |

<br>





---



- By choosing columns with the largest values and by using a mapping
  
  - {'Spiral': 0, 'Spiral_Barred': 1, 'Elliptical': 2, 'On_Edge': 3, 'Irregular': 4, 'Star/Artifact': 5}
  
  - I was able to generate a dataframe that contains `GalaxyID`, `type`, and `label`
    
    
  
  | GalaxyID | type       | label |
  | -------- | ---------- | ----- |
  | 100008   | Spiral     | 0     |
  | 100023   | Spiral     | 0     |
  | 100053   | Elliptical | 2     |
  | 100078   | Elliptical | 2     |
  | 100090   | Elliptical | 2     |
  | 100122   | Elliptical | 2     |
  | 100123   | Elliptical | 2     |
  | 100128   | Elliptical | 2     |
  | 100134   | Spiral     | 0     |
  | 100143   | On\_Edge   | 3     |



### Galaxy Images

![OG]({{'/assets/images/team39/original.png'|relative_url}})



In order to ignore the surrounding ('empty' background) pixels and solely focus on the target galaxy, I used a `CenterCrop` to crop the 180x180 pixels in the center.

![Cropped]({{'/assets/images/team39/cropped.png'|relative_url}})







## Methods / Models

### Data Augmentation to avoid overfitting

#### Rotation

![Rotation]({{'/assets/images/team39/rotation.png'|relative_url}})






#### Flip

![Flip]({{'/assets/images/team39/flipped.png'|relative_url}})





#### ColorJitter

![ColorJitter]({{'/assets/images/team39/jitter.png'|relative_url}})





#### RandomPerspective

![Perspective]({{'/assets/images/team39/perspective.png'|relative_url}})





### Model

- For Galaxy Classification, I'll finetune the pretrained ResNet18.
  
  - Achieved 80+ accuracy without image augmentation.

- For Galaxy Detection, I'll use the YOLO v3. based AstroCV from González paper.





## Final Goal

![yolo]({{'/assets/images/team39/yolo.png'|relative_url}})





Detect / classify galaxies from the R1001ED field (below, obtained from Hubble Space Telescope) and return something like above

![R1001ED]({{'/assets/images/team39/r1001ed.png'|relative_url}})





## TODO

- Try Finetuning ResNet with expanded dataset with image augmentation

- Implement YOLO based galaxy detection

- Add technical explanation/figure that shows increased accuracy after image augmentation (and/or after finetuning)

- Write a code that converts .fits file (HST R1001ED data) to RGB-channel image.

- Figure/Table Captions, replace markdown tables to images (if it looks better)





## Relevant Papers

**[1]** **Star-Galaxy Classification Using Deep Convolutional Neural Networks**

- https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.4463K/abstract
- https://github.com/EdwardJKim/dl4astro

Conventional star-galaxy classifiers use the reduced summary information from catalogues with carefully selected features (by human). The paper introduces the possibility of using deep convolutional neural networks (ConvNets) to automatically learn the features directly from the data which minimizes the input from human researchers.
<br/><br/>

[2] **Machine Learning Classification of SDSS Transient Survey Images**

- https://ui.adsabs.harvard.edu/abs/2015MNRAS.454.2026D/abstract

The paper tests performance of multiple machine learning algorithms in classifying transient imaging data from the Sloan Digital Sky Survey (SDSS) into real objects and artefacts.
<br/><br/>

[3] **Galaxy Detection and Identification Using Deep Learning and Data Augmentation**

- https://ui.adsabs.harvard.edu/abs/2018A%26C....25..103G/abstract
- https://github.com/astroCV/astroCV

The paper introduces a novel method for automatic detection and classificaiton of galaxies to make trained models more robust against the data taken from different instruments as part of AstroCV.

## References

[1] Kim, E. J. and Brunner, R. J., “Star-galaxy classification using deep convolutional neural networks”, <i>Monthly Notices of the Royal Astronomical Society</i>, vol. 464, no. 4, pp. 4463–4475, 2017. doi:10.1093/mnras/stw2672.\
[2] du Buisson, L., Sivanandam, N., Bassett, B. A., and Smith, M., “Machine learning classification of SDSS transient survey images”, <i>Monthly Notices of the Royal Astronomical Society</i>, vol. 454, no. 2, pp. 2026–2038, 2015. doi:10.1093/mnras/stv2041.\
[3] González, R. E., Muñoz, R. P., and Hernández, C. A., “Galaxy detection and identification using deep learning and data augmentation”, <i>Astronomy and Computing</i>, vol. 25, pp. 103–109, 2018. doi:10.1016/j.ascom.2018.09.004.
