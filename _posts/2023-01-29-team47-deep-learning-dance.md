---
layout: post
comments: true
title: The Influence of Deep Learning in Dance
author: Ashley Zhu
date: 2023-01-29
---

> Below is our setup for part 1...

<!--more-->

{: class="table-of-content"}

-   TOC
    {:toc}

## Project Overview

I explore the idea of computer vision in dance and utilize a convolutional neural network to predict whether the style of dance within a video can be learned through it's frames.

Check out my video walkthrough of my project [here](https://drive.google.com/drive/folders/1wT7EHr4gi3HH66cwi-gRhlWF-nWRavnt?usp=share_link)!

### AIST++ Dataset

I use the AIST++ dataset in order to grab thousands of videos of dancers doing different dance genres. Each video looks similar to the media below:

![Dancer doing Break](../assets\images\team47\project.png)

The following excerpt is needed to credit the database:

```
This project uses the AIST++ Dance Video Database cited below:

Shuhei Tsuchida, Satoru Fukayama, Masahiro Hamasaki and Masataka Goto. AIST Dance Video Database: Multi-genre, Multi-dancer, and Multi-camera Database for Dance Information Processing. In Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR 2019), 2019.

http://archives.ismir.net/ismir2019/paper/000060.pdf
```

### Keras' InceptionV3 & CNN/RNN Hybrid Model

Convolutional networks are at the core of most state-of-the-art computer vision solutions for a wide variety of tasks. Since 2014, very deep convolutional networks started to become mainstream, yielding substantial gains in various benchmarks. Although increased model size and computational cost tend to translate to immediate quality gains for most tasks (as long as enough labeled data is provided for training), computational efficiency and low parameter count are still enabling factors for various use cases such as mobile vision and big-data scenarios. Inception Architecture explores ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible by suitably factorized convolutions and aggressive regularization. We benchmark our methods on the ILSVRC 2012 classification challenge validation set demonstrate substantial gains over the state of the art: 21.2% top-1 and 5.6% top-5 error for single frame evaluation using a network with a computational cost of 5 billion multiply-adds per inference and with using less than 25 million parameters. With an ensemble of 4 models and multi-crop evaluation, we report 3.5% top-5 error on the validation set (3.6% error on the test set) and 17.3% top-1 error on the validation set. (Cite: "Rethinking the Inception Architecture for Computer Vision")

We use Keras' built-in model to train and evaluate our data for testing. This model utilizes features of both CNNs and RNNs where we structure the architecture by layers similar to CNNs while using specific unit/layer types like Gated Recurring Units that help combat the vanishing gradient problem.

### Project Code

You can check the official output of the program [here](https://drive.google.com/drive/folders/1wT7EHr4gi3HH66cwi-gRhlWF-nWRavnt?usp=share_link). The project walkthrough is located here as well as the pdf version of the code being run. The Jupyter Colab file is also located here for convenience. I referred to a video deep learning tutorial seen [here](https://github.com/AarohiSingla/Video-Classifier-Using-CNN-and-RNN/blob/main/video_classifier_working.ipynb) in order to complete my project.

#### Download Dependencies

##### AIST++ Dataset

```py
!wget https://raw.githubusercontent.com/google/aistplusplus_api/main/downloader.py
```

##### Tensorflow Docs

```py
!pip install -q git+https://github.com/tensorflow/docs
```

##### Place AIST++ Dataset Videos into AIST Folder

Only let this run up to less than 1000 videos as there could be no end to the learning and data preparation code sections if there are too many videos to go through. A future goal would be to improve the processing speed of the models for larger datasets sizes.

```py
!python downloader.py --download_folder=aist --num_processes=5
```

#### Import Packages

```py
from tensorflow_docs.vis import embed
from tensorflow import keras

import pandas as pd
import numpy as np
import tensorflow as tf

import imageio
import cv2
import re
import os
```

#### Check GPU

```py
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
```

#### Create Dataframes

A short breakdown of the data format of the AIST++ database is shown as a comment in the following code section. We want to make the data available in a more accessible format before anything, and so we utilized pandas in order to do so.

```py
'''
Data Format Rundown:

=============================================
| Dance/Music Genre   | Situations          |
=============================================
| BR: Breaking        | BM: Basic Dance     |
| PO: Pop             | FM: Advanced Dance  |
| LO: Locking         | MM: Moving Camera   |
| MH: Middle Hip Hop  | GR: Group Dance     |
| LH: LA Hip Hop      | SH: Showcase        |
| HO: House           | CY: Cypher          |
| WA: Waacking        | BT: Battle          |
| KR: Krumping        |                     |
| JS: Street Jazz     |                     |
| JB: Ballet Jazz     |                     |
=============================================

Prefixes:
{
  g: Dance Genre
  s: Situation
  c: Camera ID
  d: Dancer ID
  m: Music (type & ID)
  ch: Choreography ID
}

'''
column_names = ['File', 'Dance Genre', 'Situation', "Camera ID", "Dancer ID", 'Music Selection', 'Choreography ID']

full_df = pd.DataFrame(columns=column_names)
dataset_path = os.listdir("aist")

# Example filename: gMH_sFM_c01_d24_mMH5_ch20.mp4
video_pattern = r"g(\w{2})_s(\w{2})_c(\d{2})_d(\d{2})_m(\w{2}\d{1})_ch(\d{2}).mp4"

for item in dataset_path:
  result = re.search(video_pattern, item)

  # If the video format isn't with only one dancer, skip adding it to the dataframe
  if(len(result.groups()) == 0):
    continue
  dance_genre = result.group(1)
  situation = result.group(2)
  camera_id = result.group(3)
  dancer_id = result.group(4)
  music_selection = result.group(5)
  choreography_id = result.group(6)
  row = dict(zip(column_names, [item, dance_genre, situation, camera_id, dancer_id, music_selection, choreography_id]))
  row_df = pd.DataFrame(row, index=[item])

  full_df = pd.concat([full_df, row_df], axis=0, ignore_index=True)
```

#### Separate Train & Test Data

There is no validation dataframe because keras has a build in function to separate the test data by a fraction to validate on. This is seen later in the model building section.

```py
total_rows = len(full_df)

# 4/5 training data, 1/5 test data
train_df, test_df = np.split(full_df.sample(frac=1, random_state=42), [int(0.8* total_rows)])

print("TRAIN", train_df.head(), len(train_df))
print("TEST", test_df.head(), len(test_df))
```

#### Data Augmenting & Loading

We plan on loading the videos by extracting `max_frames` worth of frames from every video after augmenting the data by resizing and cropping it.

```py
# The following two methods are taken from this tutorial:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
IMG_SIZE = 224


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)
```

#### Feature Extraction

We use Python's build in Inception Architecture to complete feature extraction.

```py
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()
```

#### Label Encoding

This code simply extracts the labels (aka dance genres) from the dataframe.

```py
label = column_names[1]
label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(train_df[label]))
print(label_processor.get_vocabulary())

labels = train_df[label].values
labels = label_processor(labels[..., None]).numpy()
labels
```

#### Define Hyperparameters

We can play around with the hyperparameters to make the training slower or longer depending on how deeply we want to train the network.

```py
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
```

#### Prepare Train & Test Data

This section ends up taking a really long time if you downloaded a lot of videos from the command line script.

We prepare the videos for learning by first grabbing all the labels and processing them. Then, we go through all the videos and load them by extracting N number of frames from each video, as well as compressing it to a more digestible size. From there, we return the newly extracted frames and the labels associated with them.

```py
def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["File"].values.tolist()

    # take all classlabels from train_df label col and store in labels
    labels = df[label].values

    # convert class labels to label encoding
    labels = label_processor(labels[..., None]).numpy()

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool") # 145,20
    frame_features = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32") #145,20,2048

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


folder = "aist"
train_data, train_labels = prepare_all_videos(train_df, folder)
test_data, test_labels = prepare_all_videos(test_df, folder)

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")

print(f"train_labels in train set: {train_labels.shape}")
print(f"test_labels in test set: {test_labels.shape}")
```

#### Get Sequence Model

This keras model utilizes Gated Recurrent Units (common in Recurrent Neural Networks) similar to a Long Short-Term Memory Unit (LSTM) but with an output gate. These units try to solve the vanishing gradient problem that frequents RNNs. However, this is still a CNN and so our model utilizes a combination of GRU and CNN for learning.

We use include a dropout layer to regularize the data and avoid overfitting, as well as a "Dense" layer (aka Fully Connected layer) with ReLU activation to obtain the final output.

Our model utilizes cross entropy loss and ADAM optimizer to achieve learning.

```py
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model

EPOCHS = 20
# Utility for running experiments.
def run_experiment():
    filepath = "./tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


_, sequence_model = run_experiment()
```

#### Probability from Single Video

In case you only have one video that you'd like to get the probabilities on, the following functions help to return the probabilities that a video has some classification of dance genre in it.

```py
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("aist", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames

test_video = np.random.choice(test_df["File"].values.tolist())
print(f"Test video path: {test_video}")

test_frames = sequence_prediction(test_video)
```

## Three Most-Relevent Research Papers

1. #### EDGE: Editable Dance Generation From Music

    ([Code](https://arxiv.org/abs/2211.10658)) EDGE uses a transformer-based diffusion model paired with Jukebox, a strong music feature extractor, and confers powerful editing capabilities well-suited to dance, including joint-wise conditioning, and in-betweening. [1]

2. #### Dance Revolution: Long-Term Dance Generation with Music via Curriculum Learning

    ([Code](https://arxiv.org/pdf/2006.06119.pdf)) In this paper, we formalize the music-conditioned dance generation as a
    sequence-to-sequence learning problem and devise a novel seq2seq architecture to efficiently process long sequences of music features and capture the fine-grained correspondence between music and dance. [2]

3. #### The use of deep learning technology in dance movement generation
    ([Code](https://www.frontiersin.org/articles/10.3389/fnbot.2022.911469/full)) A dance movement generation algorithm based on deep learning is designed to extract the mapping between sound and motion features to solve these problems. First, the sound and motion features are extracted from music and dance videos, and then, the model is built. In addition, a generator module, a discriminator module, and a self-encoder module are added to make the dance movement smoother and consistent with the music. [3]

## References

[1] Li, Ruilong, et al. ["AI Choreographer Music Conditioned 3D Dance Generation with AIST++"](https://google.github.io/aichoreographer/)

[2] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna ["Rethinking the Inception Architecture for Computer Vision"](https://doi.org/10.48550/arXiv.1512.00567)

---
