---
layout: post
comments: true
title: CLIP and Some Applicatios
author: Ning Xu
date: 2023-01-28
---


> CLIP (Contrastive Language-Image Pre-training) is a neural network model developed by OpenAI that combines the strengths of both computer vision and natural language processing to improve image recognition and object classification.x`
CLIP has been shown to achieve state-of-the-art results on a wide range of image recognition benchmarks, and it has been used in various applications such as image captioning and image search.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Main Content in Paper
### Concepts explanation in CLIP idea
Contrastive Learning: This is a method used to teach a model to recognize the similarities and differences between different types of data. In the case of CLIP, the model is trained on pairs of images and their associated textual descriptions. During training, the model learns to identify which images and descriptions are related and which are not. This helps the model learn to understand the relationships between different types of data. 

Multi-Task Learning: Multi-task learning is a technique that allows a model to learn multiple tasks simultaneously. In the case of CLIP, the model is trained to perform multiple tasks, such as image classification, object detection, and natural language processing. This approach allows the model to learn to recognize patterns in both images and text, and to make connections between them. 

Zero-Shot Learning: Zero-shot learning is a technique used to teach a model to recognize objects or concepts that it has not been explicitly trained on. In the case of CLIP, the model is trained on a large dataset of images and their associated textual descriptions. This allows the model to learn to recognize a wide range of objects and concepts, even if it has never seen them before. 

Pre-Training on Large Datasets: CLIP is pre-trained on a large dataset of images and their associated textual descriptions. This pre-training allows the model to learn to recognize a wide range of objects and concepts, and to understand the relationships between them. This pre-training is essential to the model's ability to perform well on a wide range of tasks. Overall, these approaches work together to create a powerful model that can recognize a wide range of objects and concepts, and understand the relationships between them, using both images and text.

### The main method

The key to CLIP's ability to perform zero-shot learning is its use of a shared embedding space that maps both images and text into a common feature space. During training, CLIP is trained to associate images and their corresponding text descriptions with similar feature vectors in the embedding space, and to associate dissimilar pairs with dissimilar feature vectors. This allows CLIP to learn to recognize and differentiate between a wide variety of objects and concepts based on their textual descriptions, even if it has not been explicitly trained on them. To perform zero-shot learning, CLIP takes a new textual description as input and maps it into the embedding space to obtain a feature vector. It then searches for the image in the dataset that has the most similar feature vector to the input description. The image that best matches the input description is returned as the model's prediction. CLIP is able to perform zero-shot learning on a wide range of objects and concepts, including ones that are rare, obscure, or previously unseen, as long as they can be described using natural language. This makes it a powerful tool for a wide range of applications, including computer vision, natural language processing, and other areas where it is difficult or impractical to obtain large datasets of labeled examples.

### Image
![CLIP]({{ '/assets/images/team40/clip1.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. Learning Transferable Visual Models From Natural Language Supervision* [1].
#### Explanation
The figure shows that while standard image models train an image feature extractor and a linear classifier to predict some label, the proposed approach, CLIP, jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples.

The image encoder is a deep neural network that takes an image as input and produces a feature vector as output. The text encoder is a deep neural network that takes a text description as input and produces a feature vector as output. The two feature vectors are then compared to determine if the image and text description are a correct pairing.

At test time, the learned text encoder is used to synthesize a zero-shot linear classifier by embedding the names or descriptions of the target dataset’s classes. This means that the text encoder is used to generate a feature vector for each class in the target dataset, and these feature vectors are then used to create a linear classifier that can be used to classify images without any additional training.

#### Another attetion
The phrase "A photo of" can be considered as a visual prompt or context that is used to guide the encoding of the following text into an image embedding. By including this prompt, the CLIP model is able to understand that the text following the prompt refers to an image or a visual concept, and can use this information to generate a more accurate image embedding. 

For example, if the text following the prompt is "a cat sleeping on a windowsill", the CLIP model will use the prompt to generate an image embedding that represents the concept of a cat sleeping on a windowsill. This is done by encoding both the visual features of the cat and the windowsill, as well as the semantic meaning of the text, into a single vector representation. 

By including visual prompts such as "A photo of", the CLIP model is able to better understand the relationship between text and images, which allows it to perform tasks such as image classification and visual question answering with greater accuracy and efficiency.



### Pseudocode in paper 
```
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)
# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2

Numpy-like pseudocode for the core of an implementation of CLIP.
```
### Results of the paper
* The results of this paper show that the proposed approach is an efficient and scalable way to learn state-of-the art (SOTA) representations from scratch on large datasets. 
* After pre-training with this method, natural language can be used to reference learned visual concepts or describe new ones without needing additional labeled data - enabling zero shot transfer of the model onto downstream tasks such as OCR, action recognition in videos etc.. 
* It was benchmarked against over 30 existing computer vision datasets and it showed non trivial transfers across most tasks while being competitive with fully supervised baselines without any dataset specific training required.
* In particular, we matched the accuracy of a ResNet50 trained on ImageNet using only text descriptions instead of 1.28 million images for training purposes

## Implementaion of CLIP
The whole colab code would post in ref. The following are some explanations about the process.
### dataset
Flicker Dataset(e.g. Flicker8k_Dataset):

|             | image    |  caption     |   id |
| :---        |    :----:   |          ---: |  ---: |
| 0        | 1000268201_693b08cb0e.jpg        | A child in a pink dress is climbing up a set o...          |     0 |
| 1        | 1000268201_693b08cb0e.jpg	        | A girl going into a wooden building .   |  0 |
|2	| 1000268201_693b08cb0e.jpg| A little girl climbing into a wooden playhouse .|0|
### Hyperparameters
```
class CFG:
    debug = False
    image_path = image_path
    captions_path = captions_path
    batch_size = 32
    num_workers = 2
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1
```
### Image Encoder
```
class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """
    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
```

The code encodes each image to a fixed size vector with the size of the model's output channels (in case of ResNet50 the vector size will be 2048). This is the output after the nn.AdaptiveAvgPool2d() layer.
### Text Encoder
```
class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
```
In the case of DistilBERT (and also BERT) the output hidden representation for each token is a vector with size 768. 

### Clip Model
```
class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
```

 The code measures similarility of two groups of vectors (two matrices) are to each other with dot product (@ operator in PyTorch does the dot product or matrix multiplication in this case). To be able to multiply these two matrices together, it transposes the second one, and get a matrix with shape (batch_size, batch_size) which we will call logits. (temperature is equal to 1.0, it does not make a difference.).
### Train Model
```
 def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{CFG.captions_path}/captions.csv")
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step, epoch):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")


    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step, epoch + 1)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)
```

## Results and Some experience of hyperparameters 
TODO:
Add prompt before photo description
Size of image
Different models for text and image code

## Other Papers after clip 
### CLIP4Clip
It proposes a novel CLIP4Clip model to transfer the knowledge of an image-language pre-training model (CLIP) for video-text retrieval in an end-to-end manner.
It investigates several questions such as whether image feature alone can be used for video text retrieval and how post pretraining on large scale datasets affects performance, providing insights into practical mechanisms to better understand temporal dependency between frames in videos.

TODO


You can refer to the [source code](https://github.com/ningwebbeginner/CS188-Projects-2023Winter/tree/main/_posts) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)










## Reference
Please make sure to cite properly in your work, for example:

[1] Radford, Alec et al. “*Learning Transferable Visual Models From Natural Language Supervision*.” International Conference on Machine Learning 2021.

[2] Luo, Huaishao et al. “*CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval*.” Neurocomputing 508 (2021): 293-304.

[3] Zhang, Renrui et al. “*PointCLIP: Point Cloud Understanding by CLIP*.” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021): 8542-8552.

---
