---
layout: post
comments: true
title: Transferable Visual Models with NLP Supervision
author: Michael Simon, Victor Lin
date: 2023-03-25
---


> The manual labelling of high quality datasets for the purposes of training Computer Vision models remains one of the most time consuming tasks for Computer Vision research. Consequently, alternative methods of extracting label information from existing images and visual data is one of the areas of focus for recent research. In this blog we explore these state of the art methods in pre-training Image Classification models, namely CLIP (Contrastive Language–Image Pre-training) and CoCa (Contrastive Captioners) with a variety of pre-trainings. Extracting latent labels from images already associated with text widely available on the internet is a promising method to fast-track the training of Computer Vision models using text *and* image encoders. These models demonstrate the power of Contrastive Pre-training to perform well with "zero-shot" classification, or classifying images which the model has not been trained on or seen before.


- [Introduction](#introduction)
- [Background](#background)
    - [Contrastive Representation Learning](#contrastive-representation-learning)
    - [NLP and Zero-Shot Transfer Learning](#nlp-and-zero-shot-transfer-learning)
  - [CLIP](#clip)
    - [Encoders](#encoders)
      - [ResNets](#resnets)
      - [Vision Transformers](#vision-transformers)
      - [Text Encoder](#text-encoder)
    - [Preparing CLIP for Zero-Shot](#preparing-clip-for-zero-shot)
- [Benchmarks](#benchmarks)
  - [Model Types](#model-types)
  - [Pre-training datasets](#pre-training-datasets)
  - [OpenCLIP Benchmarks](#openclip-benchmarks)
  - [Our Zero-Shot Datasets](#our-zero-shot-datasets)
    - [Environment setup](#environment-setup)
    - [Code](#code)
    - [Miniplaces](#miniplaces)
      - [Results](#results)
- [Explainability](#explainability)
  - [Setup](#setup)
  - [Results](#results-1)
    - [General results](#general-results)
    - [Foreground Background Comparison](#foreground-background-comparison)
    - [Object Specificity Comparison](#object-specificity-comparison)
- [CoCa](#coca)
- [References](#references)
  - [LINK DUMP](#link-dump)
- [Appendix](#appendix)


# Introduction

Our project aims to examine the robustness and explainability of novel contrastive approaches to pre-training image classification models. While the original CLIP model and associated paper were released in early 2021 by OpenAI, there have already been extensive efforts done by researchers to benchmark the model's "zero-shot" classification capabilities on standards like ImageNet CITE. Nonetheless we expanded upon the existing benchmarks with more obscure datasets, including the Miniplaces dataset used this class to derive further insights into the model's capabilities. Moreover, we attempt to visualize the self-attention of the Transformer modules within CLIP models to observe how model robustness is affected by the overall architecture as well as choice of pre-training dataset. 

POTENTIALLY compare CoCa attention to CLIP (figure out hugging face stuff)


# Background

### Contrastive Representation Learning
In principle, contrastive representation learning seeks to make machine learning models more closely mimic the way humans learn to categorize the world around them. The standard approach of machine learning is to miminimize a cost function know as "loss" representing how far off a model was from predicting the correct class of input data. With contrastive learning, the loss is calculated both from the how closely the model predicted the correct class but also how "far" it was from predicting incorrect classes.

### NLP and Zero-Shot Transfer Learning
Research in 2017 revealed the potential of using Natural Language Processing (NLP) to aid in expanding the output feature space for image classification tasks. Rather than relying on a fixed set of class labels, the use of NLP allows for new labels to be introduced simply by learning a the textual encoding of said labels. This allows for zero-shot transfer learning, where models had the theoretical capability to predict classes on which they had not been trained for previously unseen data. However, the accuracy of the NLP based zero-shot method was poor on benchmark datasets such as ImageNet compared to state of the art (SoTA) methods. While the approach was improved by using a vastly larger datasets during training, training was slow and abandoned full capabilities of zero-shot transfer by restricting outputs to the output labels of the target benchmarks.

## CLIP

The Contrastive Language–Image Pre-training approach united contrastive representation learning with the existing zero-shot approach to using NLP to classify images in the form of a joint embedding matrix between text encodings and image encodings [1]. The key advancement was to use the aforementioned encodings to perform contrastive learning rather than the standard prediction where only the correct class is considered. Not only does this approach improve the training efficiency on the ImageNet accuracy benchmark compared to the status quo NLP based models, but it retains the full zero-shot capabilities. During pre-training, the model learns to maximize the similarity scores between correct image and text encodings while maximimizing dissimilirity scores between incorrect pairings:

![Combining text and image encodings](assets/images/team-33/overview-a.svg "overview-a")
*Figure 1: Summary of Contrastive Language–Image Pre-training (CLIP)*

To adopt additional labels unseen during pre-training, all that needs to be done is to learn the encoding of these new potential classes. Then, during the forward pass, the model uses the  newly augment set of text label encodings to sythesize a linear classifier which calculates the highest similarity score between input images and said labels. While the length of the text encodings does have a finite length (76 elements in the case of CLIP), this representation still allows for a broad range of labels to be introduced for almost any downstream image classification task.

![Dataset creation and Zero-shot prediction](assets/images/team-33/overview-b.svg "overview-b")
*Figure 2: Zero-shot predictive capabilities of CLIP*

### Encoders

The Image Encoder can either be a ResNet backbone or a Vision Transformer model.  Both will function properly with a given text encoder so long as they extract image features properly and project them into the latent embedding space.

#### ResNets
ResNet models are deep convolutional neural networks that use residual connections proposed by Kaiming He, which allow deep neural networks to easily learn the identity function and reduces the impact of vanishing gradients CITE. Below is an example of a simple ResNet architecture, where the curved arrows bypassing stacked 3x3 convolutions represent the residual connections.

![ResNet18 Architecture](assets/images/team-33/ResNet18.png "ResNet18")

*Figure 3: ResNet18 Architecture overview*

#### Vision Transformers

Vision Transformers take inspiration from the success of Transformer models in NLP tasks. The input image is first divided into small patches of fixed size, which are then fed into a Transformer encoder. The Transformer encoder consists of multiple layers of self-attention and feedforward neural networks, which allow the model to discover the importance of each patch and learn complex representations of the image. Finally, a classification head is added on top of the Transformer encoder to predict the class label of the image.

![ViT](assets/images/team-33/ViT.png "ViT")

*Figure 4: ViT Architecture overview*

#### Text Encoder

The Text Encoder is a variant of a BERT (Bidirectional Encoder Representations from Transformers) model, specifically BERT-base-uncased. However, discussion of the specific mechanics of the Text Encoder are beyond the scope of this blog.

### Preparing CLIP for Zero-Shot

In order to conduct zero-shot classification with a CLIP model, the model must be provided the class names for the target dataset, along with a template to convert the class names into captions. Most of the the time, the template is a basic abstraction of the original label like "A photo of a {classname}," but for other datasets the prompt may be more specific. Below is code (courtesy of LAION's CLIP_benchmark repository) to retrieve the normalized text embeddings for a given set of classnames:

```python
def zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=True, cupl=False):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.
    
    model:
        CLIP-like model with `encode_text`
    
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    classnames: list of str
        name of classes
    
    templates: list of str
        templates to use.
    
    Returns
    -------
    
    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            if cupl:
                texts = templates[classname]
            else:
                texts = [template.format(c=classname) for template in templates]
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights
```

Then during evaluation, the text encodings (referred to above as "zeroshot_weights") are multiplied with the image encodings to create probabilities for each class. Below is the code (courtesy of LAION's CLIP_benchmark repository) to run zero-shot classification on a batch of images:

```python
def run_classification(model, classifier, dataloader, device, amp=True):
    """
    Run zero-shot classifcation
    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    classifier: torch.Tensor
        obtained from the function `zero_shot_classifier`
    
    dataloader: torch.utils.data.Dataloader 
    
    Returns
    -------
    (pred, true)  where
        - pred (N, C) are the logits
        - true (N,) are the actual classes
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    pred = []
    true = []
    nb = 0
    with torch.no_grad():
        for images, target in tqdm(dataloader):
            images = images.to(device)
            target = target.to(device)

            with autocast():
                # predict
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier
            
            true.append(target.cpu())
            pred.append(logits.float().cpu())

    pred = torch.cat(pred)
    true = torch.cat(true)
    return pred, true
```

# Benchmarks

Although OpenAI released their pre-trained CLIP models to the public along with the paper publishing their findings [1], they did not publish the dataset they used to pre-train the model itself. In an effort to make public a dataset comparable to the one used by OpenAI originally, the LAION (Large-scale Artificial Intelligence Open Network) team created their own datasets of varying sizes and pretrained all the different CLIP model architectures themselves. In this section we will discuss the results from the LAION clip-benchmark repository in addition to our own benchmarks results on other datasets, including Miniplaces.

## Model Types

The model types relevant to the discussion of CLIP benchmarks are the Vision Transformer (ViT) based models, which both in the original CLIP paper and LAION benchmarks achieve higher zero-shot accuracy across the board compared to ResNet based models. The relevant models along their trainable parameter counts are:

- ViT-B/16
  - 177.35 million params (87.85 million image encoder params)
- ViT-B/32
  - 151.28 million params (86.19 million image encoder params)
- ViT-L/14
  - 427.62 million params (303.97 million image encoder params)
- ViT-H/14
  - 986.11 million params (632.08 million image encoder params)
- ViT-g/14
  - 1366.68 million params (1012.65 million image encoder params)

Here "B" refers to the "baseline" CLIP models with the smallest number of total parameters across the Image and Text encoders, while "L", "H", and "g" are shorthand "large," "huge," and "giant" CLIP models. The last number denotes the patch size used for the ViT encoder in pixels (e.g. ViT-L/14 uses a patch size of 14x14 pixels). The standard image size for most image benchmarks and past image classifcation models has been 224x224 px, which is evenly divisible by all the patch sizes above. However, there is also a version of the ViT-L/14 model which requires 336x336 input images that's included in our analysis.

## Pre-training datasets

Although the OpenAI models were trained on an a dataset never fully disclosed, the original paper details how the dataset consisted of 400 million image and text pairs scraped from the internet, typically images with associated captions [1]. The LAION created datasets of various sizes seeking to create an open source replication of the results from OpenAI's CLIP models. We'll explore the models trained on LAION 400m, a dataset of 400 million text image pairs, as well as LAION 2b with 2 billion pairs.

Note that while ViT-g/14 is the largest the researchers at LAION ran into errors during training which forced them to prematurely stop training, leaving the model to have observed a number of images comparable to that of LAION 400m. ViT-L has versions trained on both LAION 400m and LAION 2b, while and ViT-H was trained exclusively on LAION 2b.

## OpenCLIP Benchmarks

First let's briefly examine the OpenCLIP benchmarks results. The first important metric is top1 accuracy, or the percentage of samples where the model's top prediction was the correct class label.

![Dataset creation and Zero-shot prediction](assets/images/team-33/open_clip_results.png "open_clip results")
*Figure 4: OpenCLIP Top1 accuracy benchmarks*

No one model holds consistent superiority across all the VTAB (Visual Task Adaptation Benchmark) datasets along the x-axis in Figure 4. However, the larger models (ViT-L, ViT-H, and ViT-g) generally perform higher on regular classification tasks. There are some interesting spikes in variance between models in the imagenet-a dataset for instance, where the ViT-L/14 model from OpenAI far exceeds the accuracy of the ViT-L/14 from trained on LAION 2b by nearly 0.3, even though both have the exact same architecture. It even exceeded the accuracy of the ViT-g and ViT-H models, both of which are larger models. ImageNet-A consists of unmodified real-world pictures that are frequently misclassified by standard ResNet models. As both OpenAI's and LAION's datasets were scraped from image-text pairs on the internet, we can only speculate that the data used by OpenAI may have been more diverse in including images frequently misclassified by standard CNN models.

(https://paperswithcode.com/dataset/imagenet-a)

![Dataset creation and Zero-shot prediction](assets/images/team-33/averaged_models_results.png "open_clip averaged results")
*Figure 4: OpenCLIP averaged Top1 accuracy across architectures*

We can already observe contrasts in general performance for the same classification sub-task. The VTAB/flowers dataset (known better as Oxford Flowers102) and cars dataset (Stanford Cars) both test intra-class differentiation, where objects sharing a greater category (flowers and cars) must be further differentiated (flower species and car make/model/year). Here the CLIP models perform fairly well, reaching over ~70% accuracy for the Flowers102 dataset and over ~80% on the Stanford Cars dataset averaged across all models. While these metrics pale in comparison to standard classification SoTA is 99.76% for Flowers102 and 96.32% for Stanford Cars, they're competive for zero-shot. The single zero-shot data point on Papers with Code is for VAE-GAN model at [70.8% top1 accuracy](https://paperswithcode.com/sota/zero-shot-learning-on-oxford-102-flower). ViT-H/14 from LAION reaches 80.2% top1 accuracy, making it SoTA on Papers with Code. The Stanford Cars doesn't have a current leaderboard on Papers with Code, but the 80%+ zero-shot top1 accuracy surpases the SoTA for few-shot classification on Papers with Code, which is currently only [73.15%](https://paperswithcode.com/sota/few-shot-image-classification-on-stanford-2).


The zero-shot accuries of the aforementioned datasets compete with or surpass the SoTA for zero-shot and few-shot learning and are within a reasonable margin of models trained on the dataset itself.

However, the CLIP models struggle with another form of intra-class differentiation with the FGVC-Aircraft dataset. The best CLIP model was the ViT-H/14 model from LAION with 42.6% accuracy, compared to the 95.11% accuracy for Inceptionv4 which is SoTA 

https://paperswithcode.com/sota/few-shot-learning-on-fgvc-aircraft-1


Lastly, all CLIP models struggle on the dsprites datasets, all failing to surpass 10% accuracy. However, this is not surprising considering 



## Our Zero-Shot Datasets


### Tensorflow Datasets

The original CLIP paper includes a suite of benchmarks of their pretrained models on various zero shot tasks. We performed a similar benchmark analysis across a different set of datasets provided by the [Tensorflow Datsets](https://www.tensorflow.org/datasets) package. Our benchmarking differs both by including a wider diversity of datasets to test robustness, as well as including all pretrained Laion models for comparison. We make changes to key metrics from OpenCLIP that are better suited for comparing performance across datasets. In addition, to accomplish a wide breadth of datasets on a lower computational budget, we selected datasets that were smaller in size but retained much of the attributes of larger datasets:

![tfds overview](assets/images/team-33/tfds-overview.png)

The above histogram is similar to the one from the OpenCLIP benchmark results, although using completely different datsets. We make one key change to the y-axis, which uses `acc1_adjusted`. `acc1_adjusted` takes into account the random-chance accuracies of multi-class classifcation. For example, a random model on 5-class task could achieve 20% accuracy on average. As our datasets have category counts from 2-200, it is important to adjust for random chance. The computation of `acc1_adjusted` is simply:
<p style="text-align: center;">$$acc_{adjusted}\ =\ acc\ -\ \frac{1}{N}$$</p>
Where N is the number of categories. This metric enables us to better compare performance across classification tasks and avoid bias towards binary classification.

We included the imagenette and imagewang datasets as a control datasets to ensure our results align with the ones produced by OpenCLIP. Notably, imagenette and imagewang are subsets of the original imagenet and only include 10 categories. Similar to OpenCLIP's benchmarks, all clip models excel at at the imagenet tasks, acheiving >95% accuracy. This both confirms clip models ability to handle variations in imagenet tasks, but also the validity of our own benchmarks.

We can better explore CLIP's performance on high-level tasks by grouping datsets into broad types:

![tfds overview](assets/images/team-33/tfds-dataset-type.png)

As a result of the subtraction in `acc1_adjusted`, we gain the benefit of having bars that go in the negative direction, which indicate that the model performs worse than random chance. Further exploration of the tasks where negative performance occur reveals they tend have labels that are more scientific or esoteric. For example, the casava dataset contains 4 labels: `cbsd`, `cgm`, `cmd`, `healthy`, depicting different diseases inflicting a plant in the image. Besides `healthy`, all of these terms are abbreviations of diseases. The negative performance is likely a result of CLIP's dependency on the semantic meaning of labels. As these labels have no common sensical meaning, CLIP appears to have inferred counterproductive patterns from the abbreviations, resulting in negative performance. For example, `cmd` may be interpreted as the word `command` rather than the actual medical term `Cassava Mosaic Disease` in this context. Cassava belongs in the medical category, and we can see that similarly for plant and typography based tasks, CLIP has difficulty capturing the particular labeling scheme and domain knowledge of the task, resulting in poor or negative performance.

#### Code

A more detailed view of our zero-shot benchmarking can be found in this [colab](https://drive.google.com/file/d/1triv0P3MMXO3qP5bDU--Dk6xFGW_wUQO/view?usp=sharing)

### Miniplaces

We also decided it would be interesting to conduct some zero-shot analysis on Miniplaces, the toy dataset used for the class classification competition created originally by Dr. Zhou at MIT.

We constructed our [notebook](https://colab.research.google.com/drive/1Jxg_Y73J9dt42gBA9XI8qU_3fl5WDmO3?usp=sharing) using the zero-shot classifier setup from CLIP_benchmark repository from LAION. We also download the Miniplaces dataset from the google drive link in the assignments and perform the necessary unzipping setup. The template we used to convert the labels into captions was "A photo of a {classname}". After encoding the class captions, we ran zero-shot classification on the Miniplaces test set and submitted to the Kaggle competition to get top1 accuracy results.

#### Results

|Model: | ViT-L/14 LAION | ResNet50 OpenAI | ResNet101 OpenAI |
|:-----:|:-----:|:-----:|:-----:|
|Miniplaces Top1 Accuracy: | 0.726 | 0.554 | 0.555 |

The zero-shot accuracies above validate previous findings that CLIP models with ResNet based image encoders, even deep ResNets like ResNet101, fail to achieve nearly as high zero-shot accuracies as models with ViT image encoders. In the context of the class competition, which was to train on Miniplaces to evaluate standard classification accuracy, the ResNet models are still impressive, with the ResNet50 based model matching the highest standard classification accuracy and the ResNet101 surpassing all submissions. However, they're clearly not as versatile as ViT-L/14. One possible explanation for the relatively poor ResNet CLIP performance is that those architectures were designed to be good feature extractors for a finite number of classes, where the number of convolutional filters particularly in deeper layers is partially influenced by the final number of target classes. On the other hand the patchification and subsequent tokenization of images in transformer encoders are known to be better at extracting global image features and capturing long-range dependencies.


# Explainability

Now we conduct an analysis of the self-attention across different CLIP architectures and pre-trainings. Explainability is an important part of modern deep learning research, as the community moves away from treating models solely as black boxes. Similar to the class activation mappings done for ResNets in class, attention visualization attempts to visualize the sections of the image the model considers important for a given class label.

## Setup

## Results

### General results

### Foreground Background Comparison

### Object Specificity Comparison


# CoCa

In 2022 an extension of the ideas explored by CLIP was published in the form of CoCa (Contrastive Captioners). CoCa combines the contrastive learning methods of CLIP with techniques from Generative Adversarial Networks. First, note that CoCa uses a single image-text foundation that contains the to execute three different downstream tasks including Visual Recognition (single encoder), Crossmodal Alignment (dual encoder), and Image Captioning & Multimodal Understanding (encoder-decoder). However, for the purposes of this blog we will focus solely on the Image Classification capabilities of the architecture while leaving the other capabilities to a brief discussion. The dual encoder model shown below is used for downstream Image Classification.

![CoCa Overview](assets/images/team-33/coca2.jpg "overview-b")
*Figure 3: CoCa Overview*

The primary change made to the architecture is the separation of the text decoder transformer into two parts, a unimodal decoder and a multimodal decoder. For the test-time downstream image classification task, only the Unimodal Text Decoder is used. However, the model itself is trained using a combination of the contrastive loss from the image encoding and unimodal text encodings as well as the captioning loss from the multimodal text encodings and the cross  attentional pooling of image encodings. The inclusion of both the captioning loss and the contrastive loss diversifies the training process and likely allows the model to extract more semantic meaning from the latent text labels because of the added focus on captioning loss (for the purposes of generative captioning).

![Dataset creation and Zero-shot prediction](assets/images/team-33/coca.jpg "overview-b")
*Figure 4: CoCa architecture and PseudoCode*



# References

[1] Radford, Alec, et al. "Learning transferable visual models from natural language supervision." International conference on machine learning. PMLR, 2021.

[2] Yu, Jiahui, et al. "Coca: Contrastive captioners are image-text foundation models." arXiv preprint arXiv:2205.01917 (2022).

[3] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).

## LINK DUMP

https://github.com/CSAILVision/miniplaces

https://github.com/openai/CLIP/blob/main/data/prompts.md

https://github.com/moein-shariatnia/OpenAI-CLIP

https://github.com/LAION-AI/CLIP_benchmark

https://github.com/hila-chefer/Transformer-MM-Explainability/blob/main/CLIP/clip/model.py

# Appendix

| dataset                             |   ViT-B-16 laion400m_e32 |   ViT-B-16 openai |   ViT-B-16-plus-240 laion400m_e32 |   ViT-B-32 laion2b_e16 |   ViT-B-32 laion2b_s34b_b79k |   ViT-B-32 openai |   ViT-B-32-quickgelu laion400m_e32 |   ViT-H-14 laion2b_s32b_b79k |   ViT-L-14 laion2b_s32b_b82k |   ViT-L-14 laion400m_e32 |   ViT-L-14 openai |   ViT-L-14-336 openai |   ViT-g-14 laion2b_s12b_b42k |
|:------------------------------------|-------------------------:|------------------:|----------------------------------:|-----------------------:|-----------------------------:|------------------:|-----------------------------------:|-----------------------------:|-----------------------------:|-------------------------:|------------------:|----------------------:|-----------------------------:|
| cars                                |                0.836836  |         0.644572  |                         0.844174  |              0.841562  |                    0.859968  |         0.59408   |                          0.792066  |                    0.934585  |                    0.926129  |                0.895908  |         0.776893  |             0.793061  |                    0.927745  |
| country211                          |                0.181232  |         0.227962  |                         0.188436  |              0.165071  |                    0.166398  |         0.171517  |                          0.146919  |                    0.300095  |                    0.263602  |                0.230758  |         0.318246  |             0.345498  |                    0.287251  |
| fer2013                             |                0.428671  |         0.458206  |                         0.446085  |              0.477431  |                    0.469072  |         0.409446  |                          0.426721  |                    0.517554  |                    0.537058  |                0.500975  |         0.489691  |             0.480078  |                    0.465729  |
| fgvc_aircraft                       |                0.174617  |         0.241824  |                         0.185419  |              0.230423  |                    0.246325  |         0.19712   |                          0.168017  |                    0.427543  |                    0.369337  |                0.250525  |         0.317132  |             0.329433  |                    0.378038  |
| gtsrb                               |                0.434521  |         0.436263  |                         0.494854  |              0.365162  |                    0.493428  |         0.331116  |                          0.419952  |                    0.584481  |                    0.560966  |                0.5       |         0.501821  |             0.51639   |                    0.497387  |
| imagenet-a                          |                0.331867  |         0.500667  |                         0.367733  |              0.261867  |                    0.263333  |         0.313733  |                          0.216667  |                    0.592267  |                    0.5388    |                0.466133  |         0.7068    |             0.773867  |                    0.571067  |
| imagenet-r                          |                0.779     |         0.776667  |                         0.8039    |              0.759033  |                    0.764067  |         0.6932    |                          0.734167  |                    0.8932    |                    0.8741    |                0.847533  |         0.8787    |             0.891167  |                    0.8865    |
| imagenet1k                          |                0.67002   |         0.68352   |                         0.6909    |              0.65528   |                    0.665     |         0.63334   |                          0.62918   |                    0.77972   |                    0.75202   |                0.72734   |         0.7549    |             0.76548   |                    0.76664   |
| imagenet_sketch                     |                0.523001  |         0.48144   |                         0.544204  |              0.529329  |                    0.536383  |         0.423038  |                          0.493289  |                    0.665743  |                    0.632769  |                0.596141  |         0.596239  |             0.610525  |                    0.652184  |
| imagenetv2                          |                0.5958    |         0.6188    |                         0.6137    |              0.5716    |                    0.5819    |         0.5596    |                          0.5512    |                    0.7082    |                    0.6769    |                0.6559    |         0.6968    |             0.7074    |                    0.6961    |
| mnist                               |                0.6659    |         0.4955    |                         0.5699    |              0.6367    |                    0.6922    |         0.4855    |                          0.3742    |                    0.7294    |                    0.5487    |                0.764     |         0.7679    |             0.7871    |                    0.6904    |
| objectnet                           |                0.514913  |         0.554431  |                         0.538118  |              0.487671  |                    0.489986  |         0.441747  |                          0.439378  |                    0.69705   |                    0.655002  |                0.598956  |         0.690804  |             0.717778  |                    0.674653  |
| renderedsst2                        |                0.546952  |         0.611203  |                         0.578254  |              0.537068  |                    0.566172  |         0.589786  |                          0.525535  |                    0.640857  |                    0.593081  |                0.563427  |         0.698517  |             0.706755  |                    0.645799  |
| stl10                               |                0.96975   |         0.98275   |                         0.96875   |              0.9645    |                    0.96575   |         0.97075   |                          0.955     |                    0.984375  |                    0.988625  |                0.980125  |         0.99375   |             0.99425   |                    0.985875  |
| sun397                              |                0.696066  |         0.643489  |                         0.698135  |              0.684701  |                    0.686779  |         0.624667  |                          0.669649  |                    0.752156  |                    0.74332   |                0.725978  |         0.675286  |             0.686522  |                    0.754023  |
| voc2007                             |                0.768363  |         0.784388  |                         0.76262   |              0.788996  |                    0.791066  |         0.765625  |                          0.756744  |                    0.776108  |                    0.805222  |                0.755542  |         0.783253  |             0.782452  |                    0.810296  |
| vtab/caltech101                     |                0.835306  |         0.825444  |                         0.83284   |              0.831032  |                    0.83925   |         0.819198  |                          0.83284   |                    0.850427  |                    0.850427  |                0.841716  |         0.838593  |             0.837936  |                    0.852235  |
| vtab/cifar10                        |                0.9172    |         0.9083    |                         0.9272    |              0.9401    |                    0.9351    |         0.8982    |                          0.9078    |                    0.9742    |                    0.9664    |                0.9467    |         0.9569    |             0.9497    |                    0.9705    |
| vtab/cifar100                       |                0.7101    |         0.6691    |                         0.7372    |              0.7524    |                    0.7557    |         0.6453    |                          0.7021    |                    0.8468    |                    0.8336    |                0.7744    |         0.7613    |             0.7464    |                    0.8391    |
| vtab/clevr_closest_object_distance  |                0.245067  |         0.158333  |                         0.159467  |              0.1674    |                    0.1902    |         0.1716    |                          0.159333  |                    0.167667  |                    0.161     |                0.1486    |         0.160667  |             0.1582    |                    0.177267  |
| vtab/clevr_count_all                |                0.287733  |         0.2044    |                         0.231533  |              0.197933  |                    0.157333  |         0.2352    |                          0.163333  |                    0.2784    |                    0.310933  |                0.242467  |         0.189533  |             0.199     |                    0.331933  |
| vtab/diabetic_retinopathy           |                0.0873682 |         0.0633232 |                         0.131615  |              0.269955  |                    0.735411  |         0.262972  |                          0.337966  |                    0.238013  |                    0.21064   |                0.0715257 |         0.733138  |             0.732622  |                    0.434216  |
| vtab/dmlab                          |                0.150825  |         0.15905   |                         0.148581  |              0.18896   |                    0.157994  |         0.19459   |                          0.172157  |                    0.141984  |                    0.22428   |                0.185617  |         0.167495  |             0.159226  |                    0.190191  |
| vtab/dsprites_label_orientation     |                0.0303819 |         0.0197618 |                         0.0259874 |              0.0268283 |                    0.0342746 |         0.0237359 |                          0.0193685 |                    0.0260688 |                    0.0200331 |                0.0257297 |         0.023234  |             0.0243462 |                    0.0307753 |
| vtab/dsprites_label_x_position      |                0.0311008 |         0.029541  |                         0.0427924 |              0.0313314 |                    0.0302192 |         0.0346002 |                          0.0293918 |                    0.0313856 |                    0.0315348 |                0.0298394 |         0.031779  |             0.0313721 |                    0.0354004 |
| vtab/dtd                            |                0.513298  |         0.445745  |                         0.556915  |              0.540426  |                    0.557447  |         0.442553  |                          0.543085  |                    0.678723  |                    0.62766   |                0.603191  |         0.551064  |             0.556915  |                    0.681383  |
| vtab/eurosat                        |                0.503148  |         0.555926  |                         0.57963   |              0.501667  |                    0.482037  |         0.507222  |                          0.515556  |                    0.717407  |                    0.651481  |                0.617778  |         0.626667  |             0.619074  |                    0.647963  |
| vtab/flowers                        |                0.692958  |         0.713612  |                         0.710522  |              0.688567  |                    0.716051  |         0.663035  |                          0.682875  |                    0.802082  |                    0.758985  |                0.753781  |         0.791673  |             0.783542  |                    0.776061  |
| vtab/kitti_closest_vehicle_distance |                0.189873  |         0.270042  |                         0.284107  |              0.165963  |                    0.25879   |         0.274262  |                          0.288326  |                    0.111111  |                    0.229255  |                0.208158  |         0.223629  |             0.268636  |                    0.146273  |
| vtab/pcam                           |                0.605011  |         0.506287  |                         0.543701  |              0.503754  |                    0.585724  |         0.621948  |                          0.545868  |                    0.536316  |                    0.552612  |                0.485626  |         0.515991  |             0.612762  |                    0.550903  |
| vtab/pets                           |                0.892341  |         0.889343  |                         0.903788  |              0.893431  |                    0.907059  |         0.873262  |                          0.868357  |                    0.943854  |                    0.932134  |                0.918506  |         0.930771  |             0.93813   |                    0.942764  |
| vtab/resisc45                       |                0.585397  |         0.585079  |                         0.613016  |              0.61746   |                    0.60873   |         0.537619  |                          0.546032  |                    0.695714  |                    0.666667  |                0.673016  |         0.635238  |             0.637143  |                    0.717143  |
| vtab/smallnorb_label_azimuth        |                0.0590123 |         0.0558848 |                         0.0550617 |              0.0517695 |                    0.0623868 |         0.06      |                          0.0454321 |                    0.0549794 |                    0.0562963 |                0.0516872 |         0.0455967 |             0.047572  |                    0.0588477 |
| vtab/smallnorb_label_elevation      |                0.0984362 |         0.118601  |                         0.108066  |              0.108724  |                    0.114979  |         0.11786   |                          0.097284  |                    0.111276  |                    0.109465  |                0.109712  |         0.114321  |             0.112346  |                    0.113416  |
| vtab/svhn                           |                0.386063  |         0.304817  |                         0.362823  |              0.385218  |                    0.410226  |         0.123694  |                          0.278695  |                    0.561271  |                    0.463045  |                0.381761  |         0.570682  |             0.553549  |                    0.603334  |

*Table 1: Top1 Accuracy across all tested models*

*Note that the term following the model name denotes the pre-training dataset used.
*QuickGELU refers to a fast implementation of GELU (Gaussian Error Linear Unit), an activation function used mainly in Transformer architectures.


| dataset                             |   ViT-B-16 laion400m_e32 |   ViT-B-16 openai |   ViT-B-16-plus-240 laion400m_e32 |   ViT-B-32 laion2b_e16 |   ViT-B-32 laion2b_s34b_b79k |   ViT-B-32 openai |   ViT-B-32-quickgelu laion400m_e32 |   ViT-H-14 laion2b_s32b_b79k |   ViT-L-14 laion2b_s32b_b82k |   ViT-L-14 laion400m_e32 |   ViT-L-14 openai |   ViT-L-14-336 openai |   ViT-g-14 laion2b_s12b_b42k |
|:------------------------------------|-------------------------:|------------------:|----------------------------------:|-----------------------:|-----------------------------:|------------------:|-----------------------------------:|-----------------------------:|-----------------------------:|-------------------------:|------------------:|----------------------:|-----------------------------:|
| cars                                |                0.837539  |         0.646917  |                         0.845679  |              0.843564  |                    0.861549  |         0.596711  |                          0.792617  |                    0.935148  |                    0.926152  |                0.896164  |         0.777449  |             0.792772  |                    0.928858  |
| country211                          |                0.181564  |         0.228294  |                         0.188341  |              0.164218  |                    0.167014  |         0.170758  |                          0.147014  |                    0.299431  |                    0.262986  |                0.230853  |         0.317583  |             0.344597  |                    0.288009  |
| fer2013                             |                0.392124  |         0.416778  |                         0.394061  |              0.464908  |                    0.433495  |         0.35873   |                          0.398936  |                    0.505592  |                    0.533879  |                0.449919  |         0.488715  |             0.490916  |                    0.481219  |
| fgvc_aircraft                       |                0.175339  |         0.240553  |                         0.187585  |              0.23197   |                    0.246025  |         0.197344  |                          0.165811  |                    0.426096  |                    0.364929  |                0.248324  |         0.317005  |             0.331729  |                    0.378164  |
| gtsrb                               |                0.400639  |         0.370359  |                         0.431982  |              0.35121   |                    0.435373  |         0.319645  |                          0.393417  |                    0.544261  |                    0.516943  |                0.449958  |         0.439438  |             0.447213  |                    0.465594  |
| imagenet-a                          |                0.340918  |         0.483284  |                         0.381421  |              0.283902  |                    0.279051  |         0.323645  |                          0.234833  |                    0.581047  |                    0.536272  |                0.472818  |         0.67536   |             0.734575  |                    0.56392   |
| imagenet-r                          |                0.764325  |         0.760543  |                         0.790745  |              0.744452  |                    0.752173  |         0.678585  |                          0.721478  |                    0.880466  |                    0.859996  |                0.833169  |         0.865113  |             0.877613  |                    0.874816  |
| imagenet1k                          |                0.67026   |         0.68396   |                         0.69156   |              0.65632   |                    0.66506   |         0.63284   |                          0.6289    |                    0.77952   |                    0.75264   |                0.72694   |         0.7545    |             0.7656    |                    0.76656   |
| imagenet_sketch                     |                0.523367  |         0.482257  |                         0.545181  |              0.528674  |                    0.53684   |         0.42309   |                          0.494052  |                    0.665502  |                    0.632545  |                0.595678  |         0.596425  |             0.610311  |                    0.652409  |
| imagenetv2                          |                0.5956    |         0.6202    |                         0.6147    |              0.5721    |                    0.5815    |         0.5602    |                          0.5509    |                    0.709     |                    0.6781    |                0.6541    |         0.6974    |             0.7075    |                    0.6957    |
| mnist                               |                0.666533  |         0.525937  |                         0.567569  |              0.627601  |                    0.68837   |         0.457538  |                          0.370602  |                    0.733291  |                    0.543072  |                0.758934  |         0.75814   |             0.777513  |                    0.683356  |
| objectnet                           |                0.501701  |         0.536373  |                         0.527423  |              0.475086  |                    0.482038  |         0.427261  |                          0.426876  |                    0.68462   |                    0.643318  |                0.58632   |         0.673665  |             0.700719  |                    0.665057  |
| renderedsst2                        |                0.546416  |         0.611326  |                         0.57852   |              0.537306  |                    0.565867  |         0.589502  |                          0.525847  |                    0.641009  |                    0.592541  |                0.563589  |         0.69866   |             0.706832  |                    0.64594   |
| stl10                               |                0.96975   |         0.982875  |                         0.9695    |              0.965125  |                    0.966625  |         0.971875  |                          0.955375  |                    0.985     |                    0.988625  |                0.980875  |         0.993625  |             0.9945    |                    0.9865    |
| sun397                              |                0.680413  |         0.652741  |                         0.684998  |              0.678324  |                    0.684599  |         0.634691  |                          0.660945  |                    0.751282  |                    0.734839  |                0.712778  |         0.682451  |             0.692131  |                    0.752392  |
| voc2007                             |                0.803575  |         0.835061  |                         0.815719  |              0.805445  |                    0.805213  |         0.807152  |                          0.791451  |                    0.850842  |                    0.849154  |                0.830993  |         0.864352  |             0.862911  |                    0.857925  |
| vtab/caltech101                     |                0.900504  |         0.908884  |                         0.917777  |              0.903296  |                    0.909084  |         0.878652  |                          0.908529  |                    0.944071  |                    0.93943   |                0.934111  |         0.933453  |             0.932893  |                    0.944285  |
| vtab/cifar10                        |                0.9172    |         0.9082    |                         0.9272    |              0.9405    |                    0.936     |         0.8995    |                          0.9083    |                    0.9743    |                    0.9665    |                0.9466    |         0.9572    |             0.9497    |                    0.9711    |
| vtab/cifar100                       |                0.7107    |         0.6685    |                         0.7372    |              0.7529    |                    0.7554    |         0.6451    |                          0.703     |                    0.8471    |                    0.8325    |                0.7738    |         0.7611    |             0.7471    |                    0.8388    |
| vtab/clevr_closest_object_distance  |                0.166667  |         0.167624  |                         0.170202  |              0.182753  |                    0.138729  |         0.161944  |                          0.167306  |                    0.195441  |                    0.173914  |                0.144312  |         0.177231  |             0.181227  |                    0.227001  |
| vtab/clevr_count_all                |                0.282187  |         0.215112  |                         0.23329   |              0.182016  |                    0.149503  |         0.219957  |                          0.157598  |                    0.256324  |                    0.306639  |                0.23125   |         0.187025  |             0.194802  |                    0.319323  |
| vtab/diabetic_retinopathy           |                0.252036  |         0.210729  |                         0.230781  |              0.221141  |                    0.199924  |         0.219448  |                          0.259116  |                    0.233321  |                    0.23357   |                0.219634  |         0.206267  |             0.206867  |                    0.216781  |
| vtab/dmlab                          |                0.172099  |         0.170128  |                         0.148231  |              0.168168  |                    0.166126  |         0.163513  |                          0.157743  |                    0.165736  |                    0.181602  |                0.19254   |         0.1782    |             0.171342  |                    0.173338  |
| vtab/dsprites_label_orientation     |                0.0332406 |         0.0176459 |                         0.02624   |              0.0253877 |                    0.0343651 |         0.0218059 |                          0.0197744 |                    0.0268337 |                    0.0222573 |                0.0259728 |         0.0242046 |             0.0249445 |                    0.0304427 |
| vtab/dsprites_label_x_position      |                0.0321637 |         0.0292118 |                         0.0430371 |              0.0313351 |                    0.0300817 |         0.0339339 |                          0.0306791 |                    0.0307667 |                    0.032359  |                0.0308128 |         0.0324922 |             0.0320099 |                    0.0364156 |
| vtab/dtd                            |                0.509574  |         0.449468  |                         0.554255  |              0.536702  |                    0.562234  |         0.443085  |                          0.54734   |                    0.681383  |                    0.632447  |                0.604255  |         0.55      |             0.556383  |                    0.682979  |
| vtab/eurosat                        |                0.511057  |         0.546981  |                         0.58888   |              0.511475  |                    0.493914  |         0.489609  |                          0.526226  |                    0.720276  |                    0.663806  |                0.629927  |         0.638008  |             0.630934  |                    0.644557  |
| vtab/flowers                        |                0.666818  |         0.691285  |                         0.685723  |              0.673228  |                    0.6999    |         0.664526  |                          0.662814  |                    0.798549  |                    0.745564  |                0.72567   |         0.793169  |             0.786286  |                    0.781328  |
| vtab/kitti_closest_vehicle_distance |                0.256834  |         0.351792  |                         0.407683  |              0.324723  |                    0.33971   |         0.40577   |                          0.364569  |                    0.272293  |                    0.308176  |                0.179167  |         0.371703  |             0.373538  |                    0.1819    |
| vtab/pcam                           |                0.605166  |         0.506233  |                         0.543555  |              0.503552  |                    0.585768  |         0.622063  |                          0.545947  |                    0.536144  |                    0.552648  |                0.485642  |         0.515798  |             0.612803  |                    0.550852  |
| vtab/pets                           |                0.891976  |         0.884512  |                         0.903982  |              0.890606  |                    0.907089  |         0.869586  |                          0.866167  |                    0.943456  |                    0.931397  |                0.916167  |         0.93309   |             0.937208  |                    0.9434    |
| vtab/resisc45                       |                0.59327   |         0.59192   |                         0.615023  |              0.624279  |                    0.615223  |         0.541728  |                          0.554285  |                    0.706242  |                    0.675981  |                0.678134  |         0.641989  |             0.646301  |                    0.725847  |
| vtab/smallnorb_label_azimuth        |                0.0603763 |         0.0522661 |                         0.0553795 |              0.0537674 |                    0.0631907 |         0.0630119 |                          0.045024  |                    0.0559609 |                    0.0567608 |                0.05266   |         0.0456437 |             0.0457953 |                    0.0601925 |
| vtab/smallnorb_label_elevation      |                0.0977694 |         0.117643  |                         0.10851   |              0.108625  |                    0.115905  |         0.120896  |                          0.0973287 |                    0.110218  |                    0.109893  |                0.108132  |         0.113863  |             0.113338  |                    0.114651  |
| vtab/svhn                           |                0.368531  |         0.350374  |                         0.403349  |              0.379297  |                    0.421682  |         0.133264  |                          0.2796    |                    0.556531  |                    0.486993  |                0.405775  |         0.588689  |             0.559174  |                    0.568346  |
*Table 2: Mean Accuracy per Class Recall*