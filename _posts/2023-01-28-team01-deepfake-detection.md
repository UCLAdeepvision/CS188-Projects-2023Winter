---
layout: post
comments: true
title: Deepfake Detection Project Proposal
author: Janice Tsai and Zachary Chang
date: 2023-01-28
---

> Deepfake detection models and algorithms are the future in preventing malicious attacks against entities that wish to use deepfakes to circumvent modern security measures, sway the opinions of groups of people, or simply to deceive other entities. This analysis of a deepfake detection model can potentially bolster our confidence in future detection models. At the core of the effectiveness of every model is the data used to train and test the model. Good data (generally) creates better models that are more trustworthy to be deployed in real applications. In this study, the MesoNet model is used as the model for training and testing and evaluation.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## 1. Introduction & Objective
### 1.1 Introduction to Deepfakes and Detection
Deepfakes are videos of a human subject where the person's facial features are altered or swapped with another person's face in order to "change" a person's identity. One common application is videos where a person's face is exchanged for another such that it seems as though another person is speaking. The technology has been developed and improved to the point where deepfaked videos and images may be completely indistinguishable from their real counterparts. This technology is highly deceptive and may pose as a severe security, so researchers are working on deepfake detection: a method to determine whether a given sample is or is not a deepfake. In essence, deep learning has paved the way for deepfakes, but along the same vein, measures need to be created to to ensure that the technology can be limited.

### 1.2 Objective
In this study, we will be using MesoNet (see next section for more detailed information about the model) to improve deepfake detection holistically. The quality of the data used to both train and test deep learning models is vital to their ability to perform in real world scenarios where deepfake detection is crucial to security and integrity.

One primary objective of this study is to rank existing deepfake datasets and other image datasets based on their effectiveness in use with MesoNet. This is done by training and testing MesoNet on datasets and evaluating its performance using statistics. Model performance will be compared across different datasets to create a ranking for MesoNet.

Another primary objective is to then take the models that caused MesoNet to perform well and perform statistical analyses on these datasets to find commonalities or shared characteristics between them. The goal is to uncover whether there are certain traits that may be correlated to better performance with the MesoNet model. Characteristics such as diversity of the dataset (which can be measured in a variety of ways), certain facial features being present more often, or even the ethnicity of the subjects in the datasets may be considered. In this way, developers of deepfake detection models in the future may be informed of factors to be watchful of when using datasets to develop their own models.


![SampleDeepfake]({{ '/assets/images/team01/deepfakeSampleImage.jpg' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 1. Sample deepfake original (left) and deepfake (right)* [1].


## 2. About MesoNet and Model Design
### 2.1 How Deepfakes are Created
Deepfakes are generated using auto-encoders, where data is compressed using a shared encoder to reduce computational complexity. To restore the original data, the compressed data is passed through a decoder. In order to create a deepfake, an image is restored according to the decoding of the compressed version of another image.

<br>
![DeepfakeGeneration]({{ '/assets/images/team01/deepfakeGeneration.jpg' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 2. Deepfake Generation Using Encoders and Decoders* [1].

### 2.2 How MesoNet is Designed
The development of MesoNet resulted in two architectures that produced the best classification scores using a low number of parameters. In this study, we will be focusing on Meso-4. Meso-4 is designed with four convolution blocks with one hidden layer. Both ReLU and Batch Normalization are used, and Dropout is used in the fully-connected layers.

<br>
![Meso4Architecture]({{ '/assets/images/team01/meso-4-architecture.jpg' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 3. Meso-4 Architecture* [1].

For more information, please refer to <a href='https://ieeexplore.ieee.org/document/8630761'>https://ieeexplore.ieee.org/document/8630761</a>


## 3. Model Setup and Preparation
### 3.1 Environment Setup
In order to use MesoNet, the models dependencies need to be taken care of, along with the model's code. Google Colab will serve this end well with built-in editors and already pre-installed dependencies such as `pip`. The MesoNet code, authored by Darius Afchar, can be found at the link https://github.com/DariusAf/MesoNet.git.

1. Open a new Google Colab project and initialize the runtime.
2. Download the MesoNet code from GitHub and `cd` into the code directory.

    ```
    !git clone https://github.com/DariusAf/MesoNet.git
    %cd MesoNet
    ```
3. Load the dependencies of the MesoNet code into Google Colab so that each `.py` file can recognize other Python files in the same environment (there are dependencies).

    ```
    %load pipeline.py
    %load classifiers.py
    !pip3 install face_recognition
    ```

4. Try running the MesoNet code to make a prediction using the following:

    `!python example.py`

If done correctly and without modifications, the output of this should result in outputting a certain accuracy related to the pre-existing test images in the MesoNet directory. Ignore any errors that are related to videos.

Now your environment is ready to run real data.

### 3.2 Running the Model on MesoNet's Data
At this point, we want to be able to check that MesoNet indeed works with test data, which will drive the rest of this study. Sample data can be found at the link https://github.com/MalayAgr/MesoNet-DeepFakeDetection with the Google Drive link as follows: https://drive.google.com/drive/folders/15E6NZr9vhsOfX_nkOtiYkIpWZwtpNi_7.

1. Upload the data to Google Drive.
2. Access the data.
    ```
    from google.colab import drive
    drive.mount('/content/drive')
    !ls "/content/drive/My Drive/Colab Notebooks/data"
    ```
3. Position the data to be ready for consumption by MesoNet.
    ```
    # Create new data folder under /content/MesoNet
    !rm -r test_data

    !cp -r "/content/drive/My Drive/Colab Notebooks/data/test" "./"
    !mv "./test" "./test_data"
    # Rename forged folder to df
    !mv "./test_data/forged" "./test_data/df"
    !ls test_data
    ```
4. For the purposes of using this new data, we are going to modify the example Python script to now process the amount of data that is used in this trial dataset.
    ```
    import numpy as np
    from classifiers import *
    from pipeline import *
    import os, os.path
     
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
     
    # 1 - Load the model and its pretrained weights
    classifier = Meso4()
    classifier.load('weights/Meso4_DF.h5')
     
    # 2 - Minimial image generator
     
    df_path = './test_data/df'
    real_path = './test_data/real'
    df_count = len(os.listdir(df_path))
    real_count = len(os.listdir(real_path))
    data_count = df_count + real_count
    print("DATA COUNT:", data_count)
    dataGenerator = ImageDataGenerator(rescale=1./255)
    generator = dataGenerator.flow_from_directory(
            'test_data',
            target_size=(256, 256),
            batch_size=data_count,
            class_mode='binary',
            subset='training')
     
    # 3 - Predict
    X, y = generator.next()
    print('Predicted :', classifier.predict(X), '\nReal class :', y)
     
    accuracy = classifier.get_accuracy(X, y)
    print("Accuracy:", accuracy)
    # 4 - Prediction for a video dataset
     
    # classifier.load('weights/Meso4_F2F.h5')
     
    # predictions = compute_accuracy(classifier, 'test_videos')
    # for video_name in predictions:
    #     print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])
    ```
5. Test MesoNet using this new code in order to process the new test images.

    `!python example.py`

6. The output should look something similar to the following (excluding any progress bars or warnings.
    ```
    Predicted : [[0.11766662]
     [0.5160726 ]
     [0.98258865]
     ...
     [0.04207275]
     [0.9486434 ]
     [0.05716746]] 
    Real class : [0. 1. 1. ... 0. 1. 0.]
    Accuracy: [0.06616845726966858, 0.9172236323356628]
    ```

From here we can see that MesoNet has output predictions between 0 and 1 where 0 means a deepfake image while 1 means a real image. The associated true labels are listed in "Real class" and the Accuracy scores are listed below (note that for index 0, the closer to 0 means the more accurate MesoNet is with classifying deepfakes. Similarly for index 1, the closer to 1 means the more accurate MesoNet is with classifying real images).

At this point, we are ready to run our benchmark study on MesoNet using the model itself.


## 4. Proposed Study Procedure & Algorithm
In order to conduct this study, multiple datasets are required to compare the effectiveness of each dataset in training this MesoNet model. Each dataset will be tested as follows.

1. Randomly split the dataset into train and test sets (ex. 70% train - 30% test proportionally).
2. Use the `fit` function of the MesoNet classifier which calls the `train_on_batch` method to train MesoNet using the train split of the dataset.
3. Use the test set in order to check the accuracy of the model and tune hyperparameters as seen fit (create different cases of training for the learning rate, as an example, but keep these learning rates constant across all datasets).
4. Test the model's performance on the other datasets to see the accuracy of the model trained on the current dataset. 
5. Repeat for all the available datasets.
6. Rank the datasets based on their average accuracies based off several trials with possibly different hyperparameter configurations.
7. Perform analyses on the top-ranking datasets to form hypotheses for the characteristics that make up a more favorable dataset for training MesoNet (and perhaps for deepfake detection as a whole).



## References
[1] D. Afchar, V. Nozick, J. Yamagishi and I. Echizen, "MesoNet: a Compact Facial Video Forgery Detection Network," 2018 IEEE International Workshop on Information Forensics and Security (WIFS), Hong Kong, China, 2018, pp. 1-7, doi: 10.1109/WIFS.2018.8630761.