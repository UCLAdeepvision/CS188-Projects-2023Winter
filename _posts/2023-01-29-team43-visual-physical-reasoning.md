---
layout: post
comments: true
title: Visual Physical Reasoning
author: Ian Galvez, James Youn
date: 2023-01-29
---

> Our project will be based around the topic of **visual physical reasoning**. Essentially, visual physical reasoning deals with whether computers can answer "common sense" queries about an image, such as "how many chairs are at the table?", or "where is the red cube in relation to the purple ball?"

## Three Relevant Research Papers
1. ##### A simple neural network module for relational reasoning
  - [Paper] https://arxiv.org/abs/1706.01427v1
  - [Code] https://github.com/kimhc6028/relational-networks
2. ##### CLEVR-Dialog: A Diagnostic Dataset for Multi-Round Reasoning in Visual Dialog
  - [Paper] https://arxiv.org/abs/1903.03166v2
  - [Code] https://github.com/satwikkottur/clevr-dialog
3. ##### Inferring and Executing Programs for Visual Reasoning
  - [Paper] https://arxiv.org/abs/1705.03633v1 
  - [Code] https://github.com/facebookresearch/clevr-iep

## References

[1] Adam Santoro, David Raposo, David G.T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, Timothy Lillicrap (2017). A simple neural network module for relational reasoning. ArXiv. https://arxiv.org/abs/1706.01427v1

[2] Satwik Kottur, José M. F. Moura, Devi Parikh, Dhruv Batra, Marcus Rohrbach (2019). CLEVR-Dialog: A Diagnostic Dataset for Multi-Round Reasoning in Visual Dialog. ArXiv. https://arxiv.org/abs/1903.03166v2

[3] Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Judy Hoffman, Li Fei-Fei, C. Lawrence Zitnick, Ross Girshick (2017). Inferring and Executing Programs for Visual Reasoning. ArXiv. https://arxiv.org/abs/1705.03633v1

---

# Week 7 Update

### Intro and motivation
We often take for granted the way we can easily interpret visual input and understand the physical world throught that. But this task is not so simple for computers that have no preconceived notions of what the physical world is, and how images relate to physical space. **Visual physical reasoning** deals with making inferences from a given image, and demonstrating understanding by being able to answer "common sense" questions like "how many purple balls are in this frame?", or "what is to the left of the red ball?"

We chose to focus our project on visual physical reasoning because it relates a lot with the intersection of two fields that we are passionate about and believe will shape much of the future of machine learning, namely computer vision and natural language processing. Having a model that can do visual physical reasoning well would be especially useful in a setting where you want an AI to explain what is happening in an image, similar to what one might expect from a chat bot like ChatGPT.

### An example visual physical reasoning problem
These examples were taken from the following paper: [https://arxiv.org/pdf/1612.06890.pdf], which outlines the visual reasoning dataset CLEVR. It stands for "Compositional Language and Elementary Visual Reasoning", and it was created to help provide a benchmark for visual physical reasoning models to use on image datasets. Here is an example of some of the questions in the dataset, as well as example images and labels:
[Example CLEVR dataset](https://hackernoon.com/hn-images/1*oHiIMzo5XCW0mny-NGeYtw.png)

### The relational network model, and its code
Here is the baseline relational network model code that we'll be using, which we forked from the following repository: [https://github.com/satwikkottur/clevr-dialog]. 
```python
class RN(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')
        
        self.conv = ConvInputModel()
        
        self.relation_type = args.relation_type
        
        if self.relation_type == 'ternary':
            ##(number of filters per object+coordinate of object)*3+question vector
            self.g_fc1 = nn.Linear((24+2)*3+18, 256)
        else:
            ##(number of filters per object+coordinate of object)*2+question vector
            self.g_fc1 = nn.Linear((24+2)*2+18, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))


        self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)
        
        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
        
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor],2)
        

        if self.relation_type == 'ternary':
            # add question everywhere
            qst = torch.unsqueeze(qst, 1) # (64x1x18)
            qst = qst.repeat(1, 25, 1) # (64x25x18)
            qst = torch.unsqueeze(qst, 1)  # (64x1x25x18)
            qst = torch.unsqueeze(qst, 1)  # (64x1x1x25x18)

            # cast all triples against each other
            x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x26)
            x_i = torch.unsqueeze(x_i, 3)  # (64x1x25x1x26)
            x_i = x_i.repeat(1, 25, 1, 25, 1)  # (64x25x25x25x26)
            
            x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x26)
            x_j = torch.unsqueeze(x_j, 2)  # (64x25x1x1x26)
            x_j = x_j.repeat(1, 1, 25, 25, 1)  # (64x25x25x25x26)

            x_k = torch.unsqueeze(x_flat, 1)  # (64x1x25x26)
            x_k = torch.unsqueeze(x_k, 1)  # (64x1x1x25x26)
            x_k = torch.cat([x_k, qst], 4)  # (64x1x1x25x26+18)
            x_k = x_k.repeat(1, 25, 25, 1, 1)  # (64x25x25x25x26+18)

            # concatenate all together
            x_full = torch.cat([x_i, x_j, x_k], 4)  # (64x25x25x25x3*26+18)

            # reshape for passing through network
            x_ = x_full.view(mb * (d * d) * (d * d) * (d * d), 96)  # (64*25*25*25x3*26+18) = (1.000.000, 96)
        else:
            # add question everywhere
            qst = torch.unsqueeze(qst, 1)
            qst = qst.repeat(1, 25, 1)
            qst = torch.unsqueeze(qst, 2)

            # cast all pairs against each other
            x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x26+18)
            x_i = x_i.repeat(1, 25, 1, 1)  # (64x25x25x26+18)
            x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x26+18)
            x_j = torch.cat([x_j, qst], 3)
            x_j = x_j.repeat(1, 1, 25, 1)  # (64x25x25x26+18)
            
            # concatenate all together
            x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+18)
        
            # reshape for passing through network
            x_ = x_full.view(mb * (d * d) * (d * d), 70)  # (64*25*25x2*26*18) = (40.000, 70)
            
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        if self.relation_type == 'ternary':
            x_g = x_.view(mb, (d * d) * (d * d) * (d * d), 256)
        else:
            x_g = x_.view(mb, (d * d) * (d * d), 256)

        x_g = x_g.sum(1).squeeze()
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        
        return self.fcout(x_f)

```
We used this as a baseline since it's a simple enough to be a good jumping-off point for both understanding the code as well as to extend it to more generalized problems later down the line. It is notable because it makes use of the relational network model, which allows us to answer questions like, "are there any red circles that are the ***same size*** as the blue circle?" The relational network essentially is a function that keeps track of weights between certain "objects", which can be anything from the pixels on an image to whole features. We use the relational network in conjunction with an LSTM (Long Short-Term Memory) network to process natural language questions, as well as a CNN (Convolutional Neural Network) to process the image as a whole.

### Our planned experiments and extensions
To further extend our model, we want to make the model work with new training data and continue to fine-tune its parameters. Currently, the dataset of the baseline model we're working with is composed of 10,000 images and 20 questions (10 relational questions and 10 non-relational questions) per each image. Each image only consists of squares and circles that can be one of six colors. We want to expand the data set to be able to identify more shapes and colors, as well as even work with 3D datasets as well.