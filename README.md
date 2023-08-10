# Shelfhelp

This repository contains code for different experiments run and their test results in visual representation learning. Here are the list of things which will be elaborated further below.
1. Training SimCLR with ResNet50 as base model on custom dataset.
2. Randomly sampling few images to find its nearest matches from the test set.
3. Used lightly-ai, which is a computer vision framework for self-supervised learning, to train SwAV model on custom dataset and analyzed its performance.
4. Randomly selected 15 classes to understand the distribution of feature vectors on 2D plane using PCA
5. Created 7 top-level categories to understand if the products are forming clusters in 2D plane using pCA and t-SNE.

## About:
- As a part of my independent study for 2023, I worked with Shivendra Agrawal who is currently working towards developing a smartcane to help visually impaired and blind people to guide them in grocery shopping. 
- During my independent study, I trained SimCLR, SwAV models and analyzed their performance using PCA and t-SNE. In this process, I read research papers and blogs about SimCLR, SwAV, DINO, PaWS and identified that DINO and PAWS are the State of the art models for visual representation learning. I also explored a computer vision framework for self-supervised learning called lightly-ai which provides building blocks for all the SOTA models.

## Literature:
Papers:
- ![A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf)
- ![Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/pdf/2006.09882.pdf)
- ![Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/pdf/2104.14294.pdf)
- ![Semi-Supervised Learning of Visual Features by Non-Parametrically Predicting View Assignments with Support Samples](https://arxiv.org/pdf/2104.13963.pdf)

Blogs:
- https://sh-tsang.medium.com/review-simclr-a-simple-framework-for-contrastive-learning-of-visual-representations-5de42ba0bc66
- https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
- https://towardsdatascience.com/paper-explained-a-simple-framework-for-contrastive-learning-of-visual-representations-6a2a63bfa703
- https://ai.googleblog.com/2021/06/extending-contrastive-learning-to.html
- https://towardsdatascience.com/dino-emerging-properties-in-self-supervised-vision-transformers-summary-ab91df82cc3c
- https://ai.meta.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training/

Videos:
- https://www.youtube.com/watch?v=h3ij3F3cPIk
- https://www.youtube.com/watch?v=wySLC4nszv8
- https://www.youtube.com/@PyTorchLightning


## Training SimCLR model on custom dataset:
- I cloned an existing github repository for SimCLR by sthalles - ![Github repository link](https://github.com/sthalles/SimCLR.git)
- Updated the code to train Simclr model on custom dataset as the existing code is designed to train models on CIFAR/STL10.
- Dataset: Custom grocery dataset has 15269 classes 

1. Experiments:
- To understand the working of SimCLR better, I started off with training the model on 200 random classes. Below, are the training loss curve and the accuracy curve are as below.
![Accuracy curve for 200 classes for 200 epochs](simclr/images/200class_acc_curve.png?raw=true)

- Later, I finetuned the original pre-trained model with ResNet50 as base model. The accuracy curve for this is as below.
![Accuracy curve for 15269 classes for 200 epochs](simclr/images/allclass_acc_curve.png?raw=true)

2. Testing:
- For model testing purposes, I randomly selected an image, computed its similarity with all other images, and then selected the top 5 closest matches. Here are some samples of the test results.

![Sample-1](simclr/new_results/simclr_1.0_cosine/figure_33751.png)
![Sample-2](simclr/new_results/simclr_1.0_cosine/figure_5669.png)
- Though, the results look quite promising in the above images. There are few cases where the model failed to identify similar images correctly. Here are some examples.

![Sample-3](simclr/new_results/simclr_1.0_cosine/figure_20843.png)
![Sample-4](simclr/new_results/simclr_1.0_cosine/figure_42185.png)
- The outcomes demonstrate that while the model doesn't provide identical products, it is capable of recognizing products with a similar appearance, such as bottles in the initial example.

3. Deductions:
- In our scenario, SimCLR is not yielding the desired outcomes, potentially due to the extensive number of classes, approximately 15,000, and the resemblances between various products depicted in these images.
- Observing that SimCLR successfully discerns product patterns, we could consider employing it to ascertain the broader category of a product. For instance, for a product like Cheerios, its higher-level category could be identified as cereals.

## Using PCA to plot samples on 2D plane:


## Clustering of top level categories with PCA and t-SNE:
