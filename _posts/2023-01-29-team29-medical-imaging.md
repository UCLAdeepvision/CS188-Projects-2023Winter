---
layout: post
comments: true
title: medical imaging using UNet and contrastive loss
author: Charlotte Meyer and Hussein Hassan
date: 2023-1-29
---


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

<!-- ::: {.cell .markdown id="Ds10C2GR2MoY"} -->

# **Art Generation using Cycle GANS**

# Abstract

In this project, we implemented a CycleGAN to generate realistic images
of landscapes from Van Gogh paintings and vice versa. We studied the
original CycleGAN paper, as well as numerous online blogs, to gain a
deep understanding of the model\'s architecture and training process. We
then implemented the model using PyTorch and trained it on a dataset of
paired images of landscapes and Van Gogh paintings. Throughout the
project, we encountered several challenges, including limited
computational resources, which made the training process challenging.
Despite these obstacles, we were able to generate convincing Van
Gogh-style images from landscapes, although the reverse direction was
not as successful. Overall, this project provided us with an exciting
opportunity to apply our deep learning skills to a real-world problem
and to gain valuable experience working with advanced models like the
CycleGAN.

## Sources:

<https://junyanz.github.io/CycleGAN/>,
<https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix>,
<https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf>,
<https://github.com/aladdinpersson/Machine-Learning-Collection#Generative-Adversarial-Networks>
:::

<!-- ::: {.cell .markdown id="WY8vslhbMubX"} -->

# Introduction

<!-- ::: -->

<!-- ::: {.cell .markdown id="nsgrqOd_M2RU"} -->

Generative Adversarial Networks (GANs) have emerged as a powerful tool
for generating synthetic data that closely resembles real-world data.
GANs have been applied to a wide range of domains, including image and
video synthesis, natural language processing, and music generation. In
this paper, we explore the use of GANs for image-to-image translation,
specifically using CycleGANs to translate images of zebras to horses and
real-life photos to Van Gogh style paintings.

CycleGANs are a type of GAN that can learn to map images from one domain
to another domain without the need for paired data, making them
particularly useful for applications where paired data is difficult to
obtain. By using unpaired data, CycleGANs can learn to translate images
between domains that have different styles, colors, and textures.

We focus on two applications of CycleGANs. First, we explore the use of
CycleGANs to translate images of zebras to horses. This task is
particularly challenging because zebras and horses are visually similar
but have distinct differences in texture, pattern, and color. Second, we
investigate the use of CycleGANs to translate real-life photos to Van
Gogh style paintings. This task involves capturing the artistic style of
Van Gogh\'s paintings, including the use of bold brushstrokes, vivid
colors, and unique texture.

Our experiments demonstrate that CycleGANs can effectively translate
images between different domains without paired data, and can produce
high-quality, realistic images that closely resemble the target domain.
We also compare the performance of CycleGANs to other image-to-image
translation methods and discuss the advantages and limitations of using
GANs for this task.

Overall, our study highlights the potential of GANs, and in particular,
CycleGANs, for image-to-image translation, and provides insights into
the challenges and opportunities of using GANs for real-world
applications.
:::

<!-- ::: {.cell .markdown id="UB5efnETQQPc"} -->


# Configuration & Setup

<!-- ::: -->

<!-- ::: {.cell .markdown id="nDIucxORPAM-"} -->

The configuration code provided is for implementing the CycleGAN
algorithm for image-to-image translation between two domains: horse
images and zebra images. The code imports the necessary libraries,
including PyTorch, Albumentations for image augmentations, and os for
accessing files and directories. The root directory for the project and
the data directory are defined, along with the device to be used (GPU if
available, otherwise CPU). Other hyperparameters are also specified,
including the batch size, learning rate, number of workers for data
loading, and number of epochs for training. Checkpoint files for saving
and loading the trained models are also defined.

The code also defines the image transformations to be applied to the
input images during training. These transformations include resizing the
images to a fixed size, horizontal flipping with a 50% probability,
normalization of pixel values, and conversion of the images to PyTorch
tensors. An additional target \"image0\" is defined for the \"image\"
input, which is used during testing to compare the original input with
the translated output.

Overall, this configuration code sets up the necessary parameters and
transformations for training and testing a CycleGAN model on the horse
and zebra image domains, with the aim of generating realistic and
high-quality images that resemble the target domain.

<!-- :::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="8amKfr2-EFBG" outputId="7a4540a5-7451-4724-eb9f-44fb5bd04a96"} -->

```python
from google.colab import drive
drive.mount('/content/drive')
```

<!--
::: {.output .stream .stdout}
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
:::
::: -->

<!-- ::: {.cell .markdown id="QiTQRU99NhRs"} -->

## Importing Libraries

<!-- :::

::: {.cell .code id="9HhGtHjRNk70"} -->

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random, torch, os, numpy as np
import shutil
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
import copy
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
```

<!--
:::

::: {.cell .markdown id="MdJER6NeNbJg"} -->

## Global Variables

<!-- :::

::: {.cell .code id="gWNGY0syQS_m"} -->

```python
ROOT = "/content/drive/MyDrive/Colab Notebooks/CS 188 - Deep Learning/Project/CycleGAN"
DATA_ROOT = "/content/drive/MyDrive/Colab Notebooks/CS 188 - Deep Learning/Project/CycleGAN/data/vangogh2photo/vangogh2photo"

CHECKPOINT_GEN_VANGOGH = os.path.join(ROOT, "vangogh_models", "genvangogh.pth.tar")
CHECKPOINT_GEN_REALPHOTO = os.path.join(ROOT, "vangogh_models", "genrealphoto.pth.tar")
CHECKPOINT_DISC_VANGOGH = os.path.join(ROOT, "vangogh_models", "discvangogh.pth.tar")
CHECKPOINT_DISC_REALPHOTO = os.path.join(ROOT, "vangogh_models", "discrealphoto.pth.tar")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
IMG_CHANNELS = 3
LEARNING_RATE = 1e-5
LAMBDA_CYCLE = 10
NUM_WORKERS = 2
NUM_EPOCHS = 50
LOAD_MODEL = True
SAVE_MODEL = True
```

<!-- :::

::: {.cell .markdown id="K8nZdYVMU3A7"}
##Helper Functions
:::

::: {.cell .code id="hWMoDtv9U2AF"} -->

```python
def save_checkpoint(model, optimizer, filepath):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
```

<!-- :::

::: {.cell .code id="4Yi8mmosVAVw"} -->

```python

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
```

<!-- :::

::: {.cell .code id="0A3XUH2tVC3m"} -->

```python
def seed_everything(seed=188):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

<!-- :::

::: {.cell .markdown id="78BsTtERQ-q7"} -->

# Dataset

<!-- :::

::: {.cell .markdown id="bLmicP9KKwug"} -->

First we set up the dataset. We segmented the train data into train and
validation. The data was rather uneven, and was structured as follows:

1.  trainA, valA, testA is all the data of the first class
2.  trainB, valB, testB is all the data of the second class
    :::

<!-- ::: {.cell .code id="QctODL4Nkuat"} -->

```python
def make_validation():
  root_dir = "/content/drive/MyDrive/Colab Notebooks/CS 188 - Deep Learning/Project/CycleGAN"
  trainA_dir = os.path.join(root_dir, 'trainA')
  trainB_dir = os.path.join(root_dir, 'trainB')
  valA_dir = os.path.join(root_dir, 'valA')
  valB_dir = os.path.join(root_dir, 'valB')

  try:
    os.makedirs(valA_dir, exist_ok=False)
    os.makedirs(valB_dir, exist_ok=False)
  except:
    print("Validation sets already exist. If folders are empty, delete then re-run the function")
    return

  trainA_filenames = os.listdir(trainA_dir)
  trainB_filenames = os.listdir(trainB_dir)

  train_size = len(trainA_filenames)
  val_size = int(train_size * 0.1)

  # Move first val_size images from trainA to valA
  for filename in trainA_filenames[:val_size]:
      src = os.path.join(trainA_dir, filename)
      dst = os.path.join(valA_dir, filename)
      shutil.move(src, dst)

  # Move first val_size images from trainB to valB
  for filename in trainB_filenames[:val_size]:
      src = os.path.join(trainB_dir, filename)
      dst = os.path.join(valB_dir, filename)
      shutil.move(src, dst)

```

<!-- :::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="N2Zgjf7-l6jc" outputId="271fee1b-5ba2-42f0-e93c-400053794b9b"} -->

```python
make_validation()
```

<!-- ::: {.output .stream .stdout} -->

Validation sets already exist. If folders are empty, delete then re-run the function

<!-- :::
:::

::: {.cell .markdown id="WEHuXIUWNy2s"} -->

We apply transforms to the data for augmentation. The first transform is
A.Resize, which resizes the image to a fixed size of 256x256 pixels.
This ensures that all images have the same dimensions and are suitable
for input to the model.

The second transform is A.HorizontalFlip, which applies a horizontal
flip to the image with a probability of 0.5. This increases the
diversity of the training data and helps prevent overfitting.

The third transform is A.Normalize, which normalizes the pixel values
based on the mean and standard deviation of imagenet, a common practice
to ensure that the input data has similar statistical properties and
improves the convergence of the model during training. Given that the
data we were operating on was not bound to a specific type of image, it
is expected that the ImageNet statistics would provide reasonable
values.

The fourth and final transform is ToTensorV2, which converts the input
image to a PyTorch tensor. This is necessary because the CycleGAN model
is implemented in PyTorch and requires input data in tensor format.

The additional_targets parameter is used to specify that an additional
image named image0 should also be transformed in the same way as the
main image. This is used for the CycleGAN model, which requires pairs of
images to be transformed together.

Overall, these transforms are used to preprocess the input data in a way
that increases the diversity of the training data, normalizes the data,
and prepares it for input to the PyTorch-based CycleGAN model.

<!-- :::

::: {.cell .code id="jLo-ZtoCNx8f"} -->

```python
transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # not very useful
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
```

<!-- :::

::: {.cell .markdown id="MCUXa7CSPDe\_"} -->

The code provided defines a custom dataset class, HorseZebraDataset,
that extends the PyTorch Dataset class for loading and preprocessing
horse and zebra images for CycleGAN image-to-image translation. The
dataset is divided into three splits: train, validation, and test.

The init method initializes the dataset by specifying the root data
directory, split type, and image transformation. The directories for
horse and zebra images are then defined based on the split type. The
length of the dataset is set to the maximum number of images in the
horse and zebra directories. The lengths of the horse and zebra image
directories are also stored for indexing purposes.

The len method returns the length of the dataset, which is the maximum
number of images in the horse and zebra directories.

The getitem method loads a zebra and a horse image at the given index.
The index is used to ensure that the same image is not loaded multiple
times in a row. The paths to the zebra and horse images are defined
based on the index and root directories. The images are then opened
using the Pillow library (PIL) and converted to NumPy arrays. If a
transformation is provided, the images are transformed using the defined
augmentations.

Finally, the zebra and horse images are returned as a tuple. This
dataset can be used for training and testing CycleGAN models on the
horse and zebra image domains.

<!-- :::

::: {.cell .code id="Y6HZs514RTt6"} -->

```python
class VangoghRealPhotoDataset(Dataset):
    def __init__(self, root_data, split, transform=None):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.transform = transform

        self.root_vangogh = os.path.join(root_data, split + 'A')
        self.root_realphoto = os.path.join(root_data, split + 'B')

        self.vangogh_images = os.listdir(self.root_vangogh)
        self.realphoto_images = os.listdir(self.root_realphoto)

        self.length_dataset = max(len(self.realphoto_images), len(self.vangogh_images))
        self.realphoto_len = len(self.realphoto_images)
        self.vangogh_len = len(self.vangogh_images)

    def __len__(self):
        return max(len(self.realphoto_images), len(self.vangogh_images))

    def __getitem__(self, index):
        realphoto_img = self.realphoto_images[index % self.realphoto_len]
        vangogh_img = self.vangogh_images[index % self.vangogh_len]

        realphoto_path = os.path.join(self.root_realphoto, realphoto_img)
        vangogh_path = os.path.join(self.root_vangogh, vangogh_img)

        realphoto_img = np.array(Image.open(realphoto_path).convert("RGB"))
        vangogh_img = np.array(Image.open(vangogh_path).convert("RGB"))

        # realphoto_img = realphoto.transpose(2, 0, 1)
        # vangogh_img = vangogh.transpose(2, 0, 1)

        # if self.transform:
        #     realphoto_img = self.transform(realphoto_img)
        #     vangogh_img = self.transform(vangogh_img)

        if self.transform:
            augmentations = self.transform(image=realphoto_img, image0=vangogh_img)
            realphoto_img = augmentations["image"]
            vangogh_img = augmentations["image0"]

        return realphoto_img, vangogh_img
```

<!-- :::

::: {.cell .markdown id="QQ9tqb62QCCz"} -->

# Discriminator & Generator

<!-- :::

::: {.cell .markdown id="-qOegQlpUX_p"} -->

In this section we construct the Discriminator and the Generator based
on the architecture shown in the figure below:

<!-- :::

::: {.cell .markdown id="\_udnlDyDUYv8"} -->

![Architecture-of-the-generator-and-discriminator-of-unpaired-CycleGAN-Conv-2D
copy.jpg](images/f3bf0c5f2d30ff831f2b3e0af32368a3cb6b96cc.jpg)

<!-- :::

::: {.cell .markdown id="VHp12StFPYJo"} -->

The discriminator model is defined using a series of convolutional
layers to classify whether an image is real or fake. The architecture
consists of a series of convolutional blocks, with each block containing
a 2D convolution layer, an instance normalization layer, and a leaky
ReLU activation function. The initial layer takes in the input image and
reduces its size by applying a convolution with a stride of 2. The
output of the final convolutional block is passed through another
convolutional layer with a single output channel, which is then
transformed using a sigmoid activation function to produce a probability
score indicating whether the image is real or fake.

<!-- :::

::: {.cell .code id="E-lYBpbUSQtc"} -->

```python
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        # in_channels: The number of input channels to the block.
        # out_channels: The number of output channels from the block.
        # stride: The stride to use in the convolutional layer.
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
```

<!-- :::

::: {.cell .markdown id="vKlN1PDfSfUE"} -->

The Block class is used in the discriminator architecture to define
downsample blocks. Each downsample block is responsible for reducing the
spatial dimensions of the image while increasing the number of channels.
The Block class achieves this by using a convolutional layer with a
stride greater than 1, which reduces the height and width of the input
tensor while increasing the number of channels.

The Block class takes the following parameters:

- in_channels: The number of input channels to the block.
- out_channels: The number of output channels from the block.
- stride: The stride to use in the convolutional layer.

The Block class initializes a sequential module consisting of a
convolutional layer followed by instance normalization and a leaky ReLU
activation function. The Conv2d layer takes the following parameters:

- in_channels: The number of input channels to the layer.
- out_channels: The number of output channels from the layer.
- kernel_size: The size of the convolutional kernel.
- stride: The stride to use in the convolution.
- padding: The amount of padding to apply to the input.
- bias: Whether or not to include a bias term in the layer.
- padding_mode: The padding mode to use.

The InstanceNorm2d layer performs instance normalization on the output
of the convolutional layer, which normalizes the features in each
channel independently. This helps to improve the stability and
convergence of the model.

The LeakyReLU activation function introduces non-linearity into the
output of the InstanceNorm2d layer. The LeakyReLU function takes the
slope of the negative part of the function as a parameter, which is set
to 0.2 in this case.

<!-- :::

::: {.cell .code id="wPWy89-VQERM"} -->

```python
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        # in_channels: The number of input channels in the images to be classified.
        #              By default, this is set to 3 for RGB images.
        # features: A list of integers specifying the number of channels in each
        #           layer of the discriminator.
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                Block(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))
```

<!-- :::

::: {.cell .markdown id="b3rTT1FsLaaz"} -->

Next we build the generator. The code below defines the Generator model
for our CycleGAN. The chosen generator architecture is based on the
\"U-Net\" structure, which allows the model to learn a more robust
representation of the images by combining low-level and high-level
features. This is accomplished through residual layers. The exact
structure is explained below.

The Generator model consists of an initial convolutional layer that
applies a kernel of size 7 to the input image, followed by Instance
Normalization and ReLU activation. The model then has two downsample
blocks, which apply a convolutional layer with a kernel size of 3 and a
stride of 2, followed by Instance Normalization and ReLU activation.
Next, there are nine residual blocks, each consisting of two
convolutional layers with a kernel size of 3 and Instance Normalization,
with the second convolutional layer having no activation function. The
model then has two upsample blocks, which apply a transposed
convolutional layer with a kernel size of 3, a stride of 2, and output
padding of 1, followed by Instance Normalization and ReLU activation.
The final layer applies a convolutional layer with a kernel size of 7,
followed by Instance Normalization and the hyperbolic tangent activation
function.

<!-- :::

::: {.cell .code id="jCaFuaLYP3h5"} -->

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        # The ConvBlock class takes the following parameters:

        # in_channels: The number of input channels to the block.
        # out_channels: The number of output channels from the block.
        # down: A boolean value indicating whether the block should perform
        #       downsampling or upsampling. If down=True, the block performs
        #       downsampling using a nn.Conv2d layer with a stride of 2. If
        #       down=False, the block performs upsampling using a nn.ConvTranspose2d
        #       layer with a stride of 2 and an output_padding of 1 to maintain the
        #       output shape.
        # use_act: A boolean value indicating whether the block should apply a ReLU
        #          activation function after the convolutional layer.
        # **kwargs: Additional keyword arguments that are passed to the nn.Conv2d or nn.ConvTranspose2d layer.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)
```

<!-- :::

::: {.cell .code id="bAArnMbEP5zW"} -->

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # channels: The number of input and output channels to the block.
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # typical residual block structure allowing for low level features to be
        # combined with higher level features
        return x + self.block(x)
```

<!-- :::

::: {.cell .code id="Auvhu7N3QNOu"} -->

```python
class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()

        # img_channels: The number of input channels in the images to be transformed.
        # num_features: The number of channels in the initial convolutional layer
        #               and the downsample and upsample blocks.
        # num_residuals: The number of residual blocks to use in the generator.

        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features, num_features * 2, kernel_size=3, stride=2, padding=1
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features * 4,
                    num_features * 2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 1,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        self.last = nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))
```

<!-- :::

::: {.cell .markdown id="udaqjZyIRvYa"} -->

# Train

<!-- :::

::: {.cell .markdown id="-ox2UstCcilK"}
:::

::: {.cell .code id="G92-v1OeRzIJ"} -->

```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/CycleGAN')



def train_fn(
    disc_Vangogh, disc_Realphoto, gen_Realphoto, gen_Vangogh, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    # initialize counters for the real and fake Vangogh images
    Vangogh_reals = 0
    Vangogh_fakes = 0
    loop = tqdm(loader, leave=True)

    # iterate through each batch of images
    for idx, (realphoto, vangogh) in enumerate(loop):
        realphoto = realphoto.to(DEVICE)
        vangogh = vangogh.to(DEVICE)

        # Train the discriminators for vangogh and realphotos
        with torch.cuda.amp.autocast():

            # Generate fake vangogh from real
            # Classify real vangogh using disc classify the fake vangogh using disc
            # Calculate average of real vangough images classified as real, and
            # average of those classified as fake
            # Calculate the losses, then combine them

            fake_vangogh = gen_Vangogh(realphoto)
            D_Vangogh_real = disc_Vangogh(vangogh)
            D_Vangogh_fake = disc_Vangogh(fake_vangogh.detach())
            Vangogh_reals += D_Vangogh_real.mean().item()
            Vangogh_fakes += D_Vangogh_fake.mean().item()
            D_Vangogh_real_loss = mse(D_Vangogh_real, torch.ones_like(D_Vangogh_real))
            D_Vangogh_fake_loss = mse(D_Vangogh_fake, torch.zeros_like(D_Vangogh_fake))
            D_Vangogh_loss = D_Vangogh_real_loss + D_Vangogh_fake_loss


            # Do the same for the reverse process

            fake_realphoto = gen_Realphoto(vangogh)
            D_Realphoto_real = disc_Realphoto(realphoto)
            D_Realphoto_fake = disc_Realphoto(fake_realphoto.detach())
            D_Realphoto_real_loss = mse(D_Realphoto_real, torch.ones_like(D_Realphoto_real))
            D_Realphoto_fake_loss = mse(D_Realphoto_fake, torch.zeros_like(D_Realphoto_fake))
            D_Realphoto_loss = D_Realphoto_real_loss + D_Realphoto_fake_loss

            # combine the losses from both halves
            D_loss = (D_Vangogh_loss + D_Realphoto_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train the generators for vangogh and realphotos
        with torch.cuda.amp.autocast():
            # We use mse for the adversarial loss
            D_Vangogh_fake = disc_Vangogh(fake_vangogh)
            D_Realphoto_fake = disc_Realphoto(fake_realphoto)
            loss_G_Vangogh = mse(D_Vangogh_fake, torch.ones_like(D_Vangogh_fake))
            loss_G_Realphoto = mse(D_Realphoto_fake, torch.ones_like(D_Realphoto_fake))

            # cycle consistency loss here
            cycle_realphoto = gen_Realphoto(fake_vangogh)
            cycle_vangogh = gen_Vangogh(fake_realphoto)
            cycle_realphoto_loss = l1(realphoto, cycle_realphoto)
            cycle_vangogh_loss = l1(vangogh, cycle_vangogh)

            # add all togethor
            G_loss = (
                loss_G_Realphoto
                + loss_G_Vangogh
                + cycle_realphoto_loss * LAMBDA_CYCLE
                + cycle_vangogh_loss * LAMBDA_CYCLE
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_vangogh * 0.5 + 0.5, os.path.join(ROOT, f"vangogh_saved_images/vang_{idx}.png"))
            save_image(fake_realphoto * 0.5 + 0.5, os.path.join(ROOT, f"vangogh_saved_images/photo_{idx}.png"))


        loop.set_postfix(Vangogh_real=Vangogh_reals / (idx + 1), Vangogh_fake=Vangogh_fakes / (idx + 1))
```

<!-- :::

::: {.cell .markdown id="DMzoZSHRXL_5"} -->

# Running the Model

<!-- :::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":235}" id="YHtmvRT6viUI" outputId="b41adcdc-ad92-4d3e-d906-132137679e78"} -->

```python
#Initialization
disc_Vangogh = Discriminator(in_channels= IMG_CHANNELS).to(DEVICE)
disc_Realphoto = Discriminator(in_channels=IMG_CHANNELS).to(DEVICE)
gen_Realphoto = Generator(img_channels=IMG_CHANNELS, num_residuals=9).to(DEVICE)
gen_Vangogh = Generator(img_channels=IMG_CHANNELS, num_residuals=9).to(DEVICE)
```

The choice of using L1 or L2 loss in CycleGANs depends on the specific
task and the type of image transformation you are aiming to achieve.
Both L1 and L2 loss can be used in CycleGANs, and they have different
characteristics that may be more or less suitable depending on the
specific task.

**L1 loss**, also known as the mean absolute error, penalizes the
absolute difference between the predicted and target values. It is more
robust to outliers, and it tends to produce sharper images with less
blurring. On the other hand, **L2 loss**, also known as the mean squared
error, penalizes the square of the difference between the predicted and
target values. It is more sensitive to outliers, and it tends to produce
smoother images with less artifacts.

In general, L1 loss is often used in CycleGANs because it tends to
produce better results for image-to-image translation tasks, such as the
transformation of horses to zebras or real-life photos to Van Gogh-style
paintings. However, the choice between L1 and L2 loss ultimately depends
on the specific task and the trade-off between sharpness and smoothness
that is desired.

We used L1 loss (as the paper does) for the cycle consistency loss as
explained in the figure below. We also tried to test the L2 loss but due
to our limited computational resources were not able to see significant
changes without running for multiple epochs. We therefore decide to
focus our efforts on training the model with the L1 loss.

![cyclegan.png](images/45e5178fbe2f34d8a0191c63b30f0d962d1bc020.png)

We used MSE for the Adversarial loss. In the context of the CycleGAN,
the use of MSE loss as an adversarial loss has been shown to produce
better results in some cases, especially when dealing with highly
structured image domains such as maps or facial landmarks.

<!-- :::

::: {.cell .code id="5jxaMCgaXUF7"} -->

```python
opt_disc = optim.Adam(
    list(disc_Vangogh.parameters()) + list(disc_Realphoto.parameters()),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999),
)

opt_gen = optim.Adam(
    list(gen_Realphoto.parameters()) + list(gen_Vangogh.parameters()),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999),
)

L1 = nn.L1Loss() # Cycle consistency loss
mse = nn.MSELoss() # Adversarial loss
```

<!-- :::

::: {.cell .code id="RzGWJiide-\_8"} -->

```python
# Load previously trained weights
if LOAD_MODEL:
    load_checkpoint(
        CHECKPOINT_GEN_VANGOGH,
        gen_Vangogh,
        opt_gen,
        LEARNING_RATE,
    )
    load_checkpoint(
        CHECKPOINT_GEN_REALPHOTO,
        gen_Realphoto,
        opt_gen,
        LEARNING_RATE,
    )
    load_checkpoint(
        CHECKPOINT_DISC_VANGOGH,
        disc_Vangogh,
        opt_disc,
        LEARNING_RATE,
    )
    load_checkpoint(
        CHECKPOINT_DISC_REALPHOTO,
        disc_Realphoto,
        opt_disc,
        LEARNING_RATE,
    )
```

<!-- :::

::: {.cell .code id="EkDLrFdSYJFT"} -->

```python
dataset = VangoghRealPhotoDataset(
    root_data=DATA_ROOT,
    split="train",
    transform=transforms,
)

val_dataset = VangoghRealPhotoDataset(
    root_data=DATA_ROOT,
    split="val",
    transform=transforms,
)
```

<!-- :::

::: {.cell .code id="PySUstmVpUf0"} -->

```python
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True, #To increase the speed of training and avoid saving to RAM
)
g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()
```

<!-- :::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":415}" id="8ioynx-MXPs1" outputId="031be722-4da5-4379-9562-ae96b7359dbd"} -->

```python
for epoch in range(NUM_EPOCHS):
    train_fn(
        disc_Vangogh,
        disc_Realphoto,
        gen_Realphoto,
        gen_Vangogh,
        loader,
        opt_disc,
        opt_gen,
        L1,
        mse,
        d_scaler,
        g_scaler,
    )

    if SAVE_MODEL:
        save_checkpoint(gen_Vangogh, opt_gen, CHECKPOINT_GEN_VANGOGH)
        save_checkpoint(gen_Realphoto, opt_gen, CHECKPOINT_GEN_REALPHOTO)
        save_checkpoint(disc_Vangogh, opt_disc, CHECKPOINT_DISC_VANGOGH)
        save_checkpoint(disc_Realphoto, opt_disc, CHECKPOINT_DISC_REALPHOTO)
```

<!--
::: {.output .stream .stderr}
100%|██████████| 6253/6253 [18:34<00:00, 5.61it/s, H_fake=0.0838, H_real=0.905]
3%|▎ | 195/6253 [00:35<18:24, 5.49it/s, H_fake=0.0613, H_real=0.94]
::: -->

# Further Discussion

To summarize, achieving acceptable results in our experiment required a
minimum of 50 runs. We were able to run the appropriate model for 120
times, and the results can be viewed using the following link:
<https://drive.google.com/drive/folders/1rw0wNfIpGLC44rPvEfcovOzgStry19W6?usp=sharing>.

While the model effectively transforms landscapes into Van Gogh-style
images, it exhibits sub-optimal performance when transforming Van Gogh
images to real photos. This is due to the fact that Van Gogh paintings
are a lot more abstract and are a lot easier to mimic than a photo
realistic landscape, thus the task is a lot harder for a computer to
understand

**Landscape to Van Gogh**

![real_to_vangogh_44.png](images/c2eef3c55ee92d006a361a020e58971ccc12a410.png)

**Van Gogh to Landscape:**

![vangogh_to_real_25.png](images/2cb081f7d735009ef73869b1077c16b4d7b419c4.png)

<!-- ::: -->

<!-- ::: {.cell .markdown id="q5KNFutFzSBP"} -->

# Testing Data

<!-- ::: -->

<!-- ::: {.cell .code id="axweZY8ay6DF"} -->

```python
test_dataset = VangoghRealPhotoDataset(
    root_data=DATA_ROOT,
    split="test",
    transform=transforms,
)
```

<!-- ::: -->

<!-- ::: {.cell .code id="3w5Q2jfUzG4v"} -->

```python
test_loader = DataLoader(test_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
)
```

<!-- ::: -->

<!-- ::: {.cell .code id="Vg6sYIMZzRUi"} -->

```python
gen_Vangogh.eval()
gen_Realphoto.eval()
```

<!-- ::: -->

<!-- ::: {.cell .code id="mg6APFoK0JGM"} -->

```python
with torch.no_grad():
  for i, (realphoto, vangogh) in enumerate(test_loader):
    vangogh = vangogh.to(DEVICE)
    output_Realphoto = gen_Realphoto(vangogh)

    realphoto = realphoto.to(DEVICE)
    output_Vangogh = gen_Vangogh(realphoto)

    save_image(output_Realphoto * 0.5 + 0.5, os.path.join(ROOT, f"vangogh_test_saved_images/vangogh_to_real_{i}.png"))
    save_image(output_Vangogh * 0.5 + 0.5, os.path.join(ROOT, f"vangogh_test_saved_images/real_to_vangogh_{i}.png"))
```

<!-- ::: -->




---
