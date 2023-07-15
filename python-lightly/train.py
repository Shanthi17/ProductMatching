import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform, utils

from models.simclr import SimCLRModel
import argparse

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', required=True, help='path to dataset')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

args = parser.parse_args()

num_workers = 8
batch_size = args.batch_size
seed = 1
max_epochs = args.epochs
input_size = 128
num_ftrs = 32

path_to_data = args.data

transform = SimCLRTransform(input_size=input_size, vf_prob=0.5, rr_prob=0.5)

# We create a torchvision transformation for embedding the dataset after
# training
test_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ]
)

dataset_train_simclr = LightlyDataset(input_dir=path_to_data, transform=transform)

dataset_test = LightlyDataset(input_dir=path_to_data, transform=test_transform)

dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train_simclr,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)


model = SimCLRModel()
trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu")

lr_monitor = LearningRateMonitor(logging_interval="step")
model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="train_loss")
callbacks = [model_checkpoint, lr_monitor]

tb_logger = TensorBoardLogger('tb_logs', name='simsiam_tb_logs')
trainer = pl.Trainer(
    max_epochs = max_epochs,
    accelerator = 'gpu',
    enable_checkpointing = True,
    logger=tb_logger,
    callbacks=callbacks,
)


trainer.fit(model, train_dataloaders=dataloader_train_simclr, val_dataloaders=dataloader_test)
print(model_checkpoint.best_model_path)
print(model_checkpoint.best_model_score)