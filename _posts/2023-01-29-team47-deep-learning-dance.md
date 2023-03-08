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

In this post, I will explore the affect of deep learning in the field of dance and choreography. As someone who spends most of my time dancing and interacting with movement in my college career, this topic really spoke to me. I will be exploring the includion of different dance styles in the generation of choreographies.

## Project Checkpoint Code

```py
#Referencing resource @ https://www.kaggle.com/code/benenharrington/hand-gesture-recognition-database-with-cnn

import numpy as np # We'll be storing our data as numpy arrays
import os # For handling directories
from PIL import Image # For handling the images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # Plotting
```

```py
lookup = dict()
reverselookup = dict()
count = 0
 for j in os.listdir('../input/leapgestrecog/leapGestRecog/00/'):
     if not j.startswith('.'): # If running this code locally, this is to
                               # ensure you aren't reading in hidden folders
         lookup[j] = count
         reverselookup[count] = j
         count = count + 1
lookup
```

```py
x_data = []
y_data = []
datacount = 0 # We'll use this to tally how many images are in our dataset
for i in range(0, 10): # Loop over the ten top-level folders
    for j in os.listdir('../input/leapgestrecog/leapGestRecog/0' + str(i) + '/'):
        if not j.startswith('.'): # Again avoid hidden folders
            count = 0 # To tally images of a given gesture
            for k in os.listdir('../input/leapgestrecog/leapGestRecog/0' +
                                str(i) + '/' + j + '/'):
                                # Loop over the images
                img = Image.open('../input/leapgestrecog/leapGestRecog/0' +
                                 str(i) + '/' + j + '/' + k).convert('L')
                                # Read in and convert to greyscale
                img = img.resize((320, 120))
                arr = np.array(img)
                x_data.append(arr)
                count = count + 1
            y_values = np.full((count, 1), lookup[j])
            y_data.append(y_values)
            datacount = datacount + count
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size
```

```py
from random import randint
for i in range(0, 10):
    plt.imshow(x_data[i*200 , :, :])
    plt.title(reverselookup[y_data[i*200 ,0]])
    plt.show()
```

```py
import keras
from keras.utils import to_categorical
y_data = to_categorical(y_data)
```

```py
x_data = x_data.reshape((datacount, 120, 320, 1))
x_data /= 255
```

```py
from sklearn.model_selection import train_test_split
x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2)
x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)
```

```py
from keras import layers
from keras import models
```

```py
model=models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(120, 320,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

```py
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(x_validate, y_validate))
```

```py
[loss, acc] = model.evaluate(x_test,y_test,verbose=1)
print("Accuracy:" + str(acc))
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

---
