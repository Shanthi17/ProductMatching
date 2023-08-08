# Background:
## SimCLR:
It is a Simple framework for Contrastive Learning of visual Representations. The main idea behind SimCLR is to learn meaningful representations of data by maximizing the similarity between augmented views of the same image and minimizing the similarity between augmented views of different images.

Types of Augmentations used:
1. Random cropping followed by resize back to the original size
2. Random color distortions
3. Random Gaussian blur.

The network architecture consists of a base encoder like ResNet that extracts feature representations from the input images. This encoder is shared between the two augmented views of each image.

After passing the augmented views through the base encoder, a projection head is applied to map the high-dimensional feature representations to a lower-dimensional space. 

Contrastive loss:
It computes the similarity scores between pairs of augmented views and aims to make the representations of similar views closer while pushing away representations of dissimilar views. 

During training, we now update the parameters of the base encoder and the projection head to minimize contrastive loss. This helps the model capture semantic information from the data.

#### Experiments done:

#### Evaluations:
