# Retraining SimCLR model on custom dataset
I cloned an existing github repository for SimCLR by sthalles - ![Github repository link](https://github.com/sthalles/SimCLR.git)

Updated the code to train Simclr model on custom dataset as the existing code is designed to train models on CIFAR/STL10.

### Dataset: Custom grocery dataset has 15269 classes 

## Experiments:
To understand the working of SimCLR better, I started off with training the model on 200 random classes. Below, are the training loss curve and the accuracy curve are as below.
![Accuracy curve for 200 classes for 200 epochs](images/200class_acc_curve.png?raw=true)

Later, I finetuned the original pre-trained model with ResNet50 as base model. The accuracy curve for this is as below.
![Accuracy curve for 15269 classes for 200 epochs](images/allclass_acc_curve.png?raw=true)

## Testing:

For model testing purposes, I randomly selected an image, computed its similarity with all other images, and then selected the top 5 closest matches. Here are some samples of the test results.

![Sample-1](new_results/simclr_1.0_cosine/figure_33751.png)
![Sample-2](new_results/simclr_1.0_cosine/figure_5669.png)

Though, the results look quite promising in the above images. There are few cases where the model failed to identify similar images correctly. Here are some examples.

![Sample-3](new_results/simclr_1.0_cosine/figure_20843.png)
![Sample-4](new_results/simclr_1.0_cosine/figure_42185.png)

The outcomes demonstrate that while the model doesn't provide identical products, it is capable of recognizing products with a similar appearance, such as bottles in the initial example.

## Deductions:
1. In our scenario, SimCLR is not yielding the desired outcomes, potentially due to the extensive number of classes, approximately 15,000, and the resemblances between various products depicted in these images.
2. Observing that SimCLR successfully discerns product patterns, we could consider employing it to ascertain the broader category of a product. For instance, for a product like Cheerios, its higher-level category could be identified as cereals.


Readme file for the original repo: ![sthalles-README](https://github.com/sthalles/SimCLR/blob/master/README.md)
