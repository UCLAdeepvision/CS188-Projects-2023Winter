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
* TOC
{:toc}

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
