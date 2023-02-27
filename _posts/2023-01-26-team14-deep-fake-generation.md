---
layout: post
comments: true
title: Deep Fake Generation
author: Sarah Mauricio and Andres Cruz
date: 2023-01-26
---

## Abstract

> The use of deep learning methods in deep fake generation has contributed to the rise of fake of fake images which has some very serious ethical dilemmas. We will look at two different ways to generate deepfake pictures and videos, and will then focus in on Image-to-Image Translation. CycleGAN and StarGAN are two different models we will by studying to create deep fake images using Image-to-Image Translation.
<!--more-->
## Table of Contents

* [Introduction](#intro)
* [What is Deepfake](#deepfake)
    * [Example: Image-to-Image Translation](#i2i)
    * [Example: Image Animation](#ia)
* [What is a Generative Adversarial Network (GAN)](#gan)
* [Cycle GAN](#cyclegan)
    * [Motivation](#mot1)
    * [Architecture](#arch1)
    * [Architecture Blocks and Code Implementation](#archblocks1)
    * [Results](#res1)
* [Star GAN](#stargan)
    * [Motivation](#mot2)
    * [Architecture](#arch2)
    * [Architecture Blocks and Code Implementation](#archblocks2)
    * [Results](#res2)
* [Demo](#demo)
* [References](#ref)

## Introduction <a name="intro"></a>

In this post, we will be covering two different models used for image-to-image translation. These models can be used for creating Deepfake media.

## What is Deepfake <a name="deepfake"></a>

Deepfake is a term used to describe artificially constructed media that portrays an individual or individuals in a way that suits the creator. For example, creating image to image translations, image animations, audio reconstruction, and more. Deepfakes are created using deep neural networks architectures, such as Generative Adversarial Networks or Autoencoders. 

### Example: Image-to-Image Translation <a name="i2i"></a>

Image to image translation is the process of extracting features from a source image and emulating those features in another image. An example would be Neural Style Transfer, where a source image is used to create an art style to transfer to another image.

![Style Transfer](/assets/images/team14/style_transfer.png)
* Fig 1. Example of Neural Style Transfer (Image source: https://www.v7labs.com/blog/neural-style-transfer)

Below are some common architectures seen for image to image translation.

![image to image](/assets/images/team14/image-to-image.png)
* Fig 2. Example of image to image translation architectures (Image source: https://www.researchgate.net/publication/366191121_Enhancing_cancer_differentiation_with_synthetic_MRI_examinations_via_generative_models_a_systematic_review)

### Example: Image Animation <a name="ia"></a>

Image Animation is the action of generating a video where the object from an image is animated using the action from a driving video. For example, if we had an image of a water bottle and a driving video of a ball flying across the screen, the output video would be a water bottle flying across the screen. Thus, it will create an animation based on a single image.

![GAN Flow](/assets/images/team14/pipeline.png)
* Fig 3. Example flow of Image Animation

Once applying the model, we would see results similar to the following:

![Image Animation Output](/assets/images/team14/vox-teaser.gif)
* Figure 4. Example output from Image Animation


## What is a Generative Adversarial Network (GAN) <a name="gan"></a>

Generative Adversarial Network, or GAN, is the core framework behind a lot of the DeepFake algorithms you may come across. It is an approach to generate a model for a dataset using deep learning priciples. Generative modeling automatically discovers and learns the patterns in the data so that the model can be used to generate new images that could have been a part of the original dataset. GANs train a generative model that consists of two sub-components: the generator models which is trained to generate new images and the discriminator model which tries to classify an image as real or fake. The generative models and the discriminator model are trained together in an adversarial way, meaning until the discrimnator model classifies images incorrectly about half of the time. This would mean that the generator model generates DeepFake images that could pass as being real.

![GAN Flow](/assets/images/team14/gan1.JPG)
* Fig 5. Example of GAN Flow

Below we look into two different models using ideas from GAN.

## Cycle GAN <a name="cyclegan"></a>

### Motivation <a name="mot1"></a>

![unpaired images](/assets/images/team14/unpaired-images.webp)
* Fig 6. Example of paired and unpaired images, (Image source: https://towardsdatascience.com/cyclegan-learning-to-translate-images-without-paired-training-data-5b4e93862c8d)

CycleGAN was used in order to use unpaired image to image translations rather than paired image to image translations. This would allow for more training data and more robust outputs for translations. This model seems to work well on tasks that involve color or texture changes, like day-to-night photo translations, or photo-to-painting tasks like collection style transfer. However, tasks that require substantial geometric changes to the image, such as cat-to-dog translations, usually fail [insert citation].

### Architecture <a name="arch1"></a>


The architecture of CycleGAN consists of a generator taken from Johnson et al [3], which consists of 3 convolutional layers, 6 residual block layers, 2 transpose convolutional layers and a final convolution output layer. Should also be noted that all layers similar to Johnson et al are followed by instance normalization.

![CycleGAN Generator](/assets/images/team14/cycleGAN-generator.png)
* Fig 7. Example of CycleGAN Generator architecture (Image source: https://towardsdatascience.com/cyclegan-learning-to-translate-images-without-paired-training-data-5b4e93862c8d)

The discriminator uses a 70x70 PatchGAN architecture, which are used to classify 70x70 overlapped images to see if they are real or fake. The PatchGAN architecture consists of 5 convolutional layers with instance normalization [5].

![CycleGAN Discriminator](/assets/images/team14/cycleGAN-discriminator.webp)
* Fig 8. Example of CycleGAN Discriminator architecture (Image source: https://towardsdatascience.com/cyclegan-learning-to-translate-images-without-paired-training-data-5b4e93862c8d)

The complete model consists of two Generators and two Discriminators. Each Generator/Discriminator pair tries to map an image from one domain to another while the other pair tries to map the reverse of the image.
![CycleGAN Complete Model](/assets/images/team14/CycleGAN-complete.jpg)
*  Fig 9. CycleGAN architecture with all generators and discriminators. (Image source: https://cvnote.ddlee.cc/2019/08/21/image-to-image-translation-pix2pix-cyclegan-unit-bicyclegan-stargan)
 
 
#### Loss Functions

There are two kinds of loss in the CycleGAN architecture. There is the normal adversarial loss that we typically associate with GANs and there is the cyclic loss that is used in the CycleGAN implementation. Since the architecture contains two GAN networks, the mapping functions are from $G : X \to Y$ and $F : Y \to X$ where the discriminators are $D_Y$ and $D_X$ respectively. G will map images from the X domain to the Y domain while $D_Y$ will try to discriminate the images that G mapped. F will do the same thing however, it will do it from the Y domain over to the X domain with $D_X$ trying to discriminate the mapped images.

Log likelihood loss was used for the adversarial loss in Zhu et al's implementation [7]. The loss function for the adversarial loss is as follows:

$$
L_{GAN}(G, D_Y, X, Y)=\mathbb{E}_{y\sim p_{data}(y)}[log\hspace{0.1cm}D_Y(y)]+\mathbb{E}_{x\sim p_{data}(x)}[log(1-D_Y(G(x)))]
$$

This loss function only takes into account the G mapping from X domain to Y domain. In order to optimize loss for the other network, another adversarial loss function would need to be used for the F mapping. So the total loss would need to include two sets of adversarial loss. 

Then there is also the cyclic-consistency loss which ensures that $F(G(x)) \approx x$ and $G(F(y)) \approx y$. L1 norm loss was used for the cyclic loss function.

$$
L_{cyc}(G, F)=\mathbb{E}_{x\sim p_{data}(x)}[||F(G(x))-x||_1] + \mathbb{E}_{y\sim p_{data}(y)}[||G(F(y))-y||_1]
$$

The complete objective function is as follows:

$$
L(G, F, D_X, D_Y) = L_{GAN}(G,D_Y,X,Y) + L_{GAN}(G,D_Y,X,Y) + \lambda L_{cyc}(G,F)
$$

where $\lambda$ controls the importance between the two types of losses.

According to Zhu et al [7], they aim to solve: 

$$
G^{\ast}, F^{\ast} = arg\ \underset{G,F}{min}\ \underset{D_X,D_Y}{max}\ L(G, F, D_X, D_Y)
$$


### Architecture Blocks and Code Implementation <a name="archblocks1"></a>
The following code blocks for the CycleGAN implementation were taken from [junyanz/pytorch-CycleGAN-and-pix2pix repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/models).

The code below shows the complete implementation of the CycleGAN model. However, the important thing to note is that the the network architecture is created by calling the define_G and define_D functions which instantiate a generator and discriminator model respectively. The CycleGan model then proceeds to define the optimizers and criterion for the generators and discriminators.
```
class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.
    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
```

When the CycleGAN model calls the function define_G(), the model accesses the ResnetGenerator class which creates the generator for the CycleGAN model.
```
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
```
This is an implementation of the ResnetBlocks used in the ResnetGenerator model. Which just consists of a convolutional layer, a normalization layer and a ReLU activation layer.
```
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
```

The discriminator model used for the CycleGAN is a PatchGAN implementation. The define_D function in the CycleGAN model accesses this class and creates a PatchGAN.
```
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
```



### Results <a name="res1"></a>

## Star GAN <a name="stargan"></a>

### Motivation <a name="mot2"></a>
StarGAN is a generative adversarial network that learns the mappings among multiple domains using only a single generator and a discriminator, training effectively from images of all domains (Choi 2). The topology could be represented as a star where multi-domains are connected, thus receiveing the name StarGAN. 

![StarGAN Results](/assets/images/team14/star1.JPG)
* Fig 10. Example of multi-domain image-to-image translation on CelebA dataset using StarGAN

StarGAN consists of two modules, a discriminator and a generator. The discriminator learns to differentiate between real and fake images and begins to clssify the real images with its proper domain. The generator takes an image and a target domain label as input and generates a fake image with them. The target domain label is then spatially replicated and concatenated with the image given as input. The generator attempts to reconstruct the orginal image via the fake image when given the original domain label. Lastly, the generator tries to generate images that are almost identical to the real images and will be classified as being from the target domain by the discriminator.

![StarGAN Flow](/assets/images/team14/star2.JPG)
* Fig 11. Example flow of StarGAN where D represents the discriminator and G represents the generator

The overarching goal of StarGAN is to translate images from one domain to the other domain. For example, translating an image with a red leaves to an image with yellow leaves.

### Architecture <a name="arch2"></a>

The architecture of StarGAN consists of a generator, which consists of two convolutional layers with a stride size of two for downsampling, six residual blocks, and two tranposed convolutional layers with a stride size of two for upsampling. Instance normalization is also used in all layers except the last. The architecture we use for this is an adaptation of the CycleGAN generator.

In both this image and the next, N is the number of output channels, K is the kernel size, S is the stride sie, P is the padding size, IN is the instance normalization, n_d is the number of the domain, and n_x is the dimension of the domain labels.

![StarGAN Generator](/assets/images/team14/star4.JPG)
* Fig 12. Example of StarGAN Generator Architecture (Image source: https://arxiv.org/pdf/1711.09020v3.pdf)

The discriminator uses a single convolutional layer for the input layer, then 5 hidden convolutional layers, then 2 convolutional output layers. It uses Leaky ReLU with a negative slope of 0.01. This stride size is 2 for the input and hidden layers, and the stride is 1 in the output layers.

![StarGAN Discriminator](/assets/images/team14/star5.JPG)
* Fig 13. Example of StarGAN Discriminator architecture (Image source: https://arxiv.org/pdf/1711.09020v3.pdf)

#### Loss Functions

The discriminator produces probability distributions over source and domain labels as follows:

$$
\mathbf{D} : \mathbf{x}\rightarrow\{{D_{src}(x), D_{cls}(x)}\}
$$

To ensure the generated images are indistinguishable from the real images, adversarial loss is used:

$$
\mathbf{L}_{adv} = \mathbb{E}_{x}[log D_{src}(x)] + \mathbb{E}_{x,c}[log (1-D_{src}(G(x,c)))]
$$

Here, G generates an image G(x,c) that is conditioned on the input image, x, and the target domain label, c. D then tries to determine if it is a real or fake image. 

We need to add an auxiliary classifier on top of our D and impose the domain classification loss when optimizing D and G in order to classify the output image y that was generated from our input image dx.

$$
\mathbf{L}^{r}_{cls} = \mathbb{E}_{x, c'}[-log D_{cls}(c'|x)]
$$

Then, our loss function for domain classification is:

$$
\mathbf{L}^{f}_{cls} = \mathbb{E}_{x, c}[-log D_{cls}(c|G(x,c))]
$$

Now, out adversarial and classification losses are minimized, but this does not guarantee that the translated images preserve the content of its imput images. To reduce this issue, we aneed to apply a cycle consistency loss to G:

$$
\mathbf{L}_{rec} = \mathbb{E}_{x, c, c'}[||x-G(G(x, c), c')||]
$$

Thus, the to optimize G and D we have the following formulas:

$$
\mathbf{L}_{D} = -\mathit{L}_{adv} + {\lambda}_{cls}\mathit{L}^{r}_{cls}
$$

$$
\mathbf{L}_{G} = \mathit{L}_{adv} + {\lambda}_{cls}\mathit{L}^{f}_{cls} + {\lambda}_{rec}\mathit{L}_{rec}
$$

Lastly, we improve our loss function to generate higher quality images and to stabilize the trianing process. The new loss formula uses Wasserstein's GAN objective with gradient penalty:

$$
\mathbf{L}_{adv} = \mathbb{E}_{x}[D_{src}(x)] - \mathbb{E}_{x,c}[D_{src}(G(x,c))] - {\lambda}_{gp}\mathbb{E}_{\hat{x}}[(||{\triangledown}_\hat{x}\mathit{D}_{src}(\hat{x})||_{2}-1)^{2}]
$$

### Architecture Blocks and Code Implementation <a name="archblocks2"></a>

This is the ResidualBlock module. It consists of the Conv2D, instance norm, and ReLu, which are all modules in PyTorch. The goal of this module is to ensure the neural network is able to expand in depth without errors occuring during backpropgation.

```
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
```

The code for the Generator is as follows. It uses the ResidualBlock module that is defined above. 

```
class GeneratorResNet(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), res_blocks=9, c_dim=5):
        super(GeneratorResNet, self).__init__()
        channels, img_size, _ = img_shape

        # Initial convolution block
        model = [
            nn.Conv2d(channels + c_dim, 64, 7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        curr_dim = 64
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim *= 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(curr_dim)]

        # Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim = curr_dim // 2

        # Output layer
        model += [nn.Conv2d(curr_dim, channels, 7, stride=1, padding=3), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, c), 1)
        return self.model(x)
```

The code for the Discriminator is as follows.

```
class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), c_dim=5, n_strided=6):
        super(Discriminator, self).__init__()
        channels, img_size, _ = img_shape

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1), nn.LeakyReLU(0.01)]
            return layers

        layers = discriminator_block(channels, 64)
        curr_dim = 64
        for _ in range(n_strided - 1):
            layers.extend(discriminator_block(curr_dim, curr_dim * 2))
            curr_dim *= 2

        self.model = nn.Sequential(*layers)

        # Output 1: PatchGAN
        self.out1 = nn.Conv2d(curr_dim, 1, 3, padding=1, bias=False)
        # Output 2: Class prediction
        kernel_size = img_size // 2 ** n_strided
        self.out2 = nn.Conv2d(curr_dim, c_dim, kernel_size, bias=False)

    def forward(self, img):
        feature_repr = self.model(img)
        out_adv = self.out1(feature_repr)
        out_cls = self.out2(feature_repr)
        return out_adv, out_cls.view(out_cls.size(0), -1)
```



### Results <a name="res2"></a>

## Demo <a name="demo"></a>

## References <a name="ref"></a>

[1] Brownlee, Jason. “A Gentle Introduction to Generative Adversarial Networks (Gans).” MachineLearningMastery.com, 19 July 2019, https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/.

https://github.com/Deepfakes/

[2] Goodfellow, Ian J., et al. Generative Adversarial Networks. 2014. DOI.org (Datacite), https://doi.org/10.48550/ARXIV.1406.2661.

https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py

[3] Johnson, Justin, et al. Perceptual Losses for Real-Time Style Transfer and Super-Resolution. arXiv, 26 Mar. 2016. arXiv.org, https://doi.org/10.48550/arXiv.1603.08155.

[4] Tolosana, Ruben, et al. DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection. 2020. DOI.org (Datacite), https://doi.org/10.48550/ARXIV.2001.00179.

https://github.com/deepfakes/faceswap

[5] Wolf, Sarah. “CycleGAN: Learning to Translate Images (Without Paired Training Data).” Medium, 20 Nov. 2018, https://towardsdatascience.com/cyclegan-learning-to-translate-images-without-paired-training-data-5b4e93862c8d.

[6] Zhang, Tao. “Deepfake Generation and Detection, a Survey.” Multimedia Tools and Applications, vol. 81, no. 5, Feb. 2022, pp. 6259–76. DOI.org (Crossref), https://doi.org/10.1007/s11042-021-11733-y.

[7] Zhu, Jun-Yan, et al. Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks. arXiv, 24 Aug. 2020. arXiv.org, https://doi.org/10.48550/arXiv.1703.10593.

[8] “Pytorch-CycleGAN-and-Pix2pix/Models at Master · Junyanz/Pytorch-CycleGAN-and-Pix2pix.” GitHub, https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix. Accessed 26 Feb. 2023.
