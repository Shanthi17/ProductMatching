# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.data import LightlyDataset

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import argparse
import os 

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--name', required=True, help="type of model being used")
parser.add_argument('--data', metavar='DIR', required=True, help='path to dataset')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')

args = parser.parse_args()

class SimCLR(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        # Freeze the weights
        for param in resnet.parameters():
            param.requires_grad = False
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead(512, 2048, 2048)
        self.criterion = NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim


model = SimCLR()

transform = SimCLRTransform(input_size=32)
# dataset = torchvision.datasets.CIFAR10(
#     "datasets/cifar10", download=True, transform=transform
# )
# or create a dataset from a folder containing images or videos:
dataset = LightlyDataset(args.data, transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"

# trainer = pl.Trainer(max_epochs=args.epochs, devices=1, accelerator=accelerator)

lr_monitor = LearningRateMonitor(logging_interval="step")
model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="train_loss")
callbacks = [model_checkpoint, lr_monitor]

tb_logger = TensorBoardLogger('tb_logs', name=f'{args.name}_tb_logs')
trainer = pl.Trainer(
    max_epochs = args.epochs,
    devices=1,
    accelerator = accelerator,
    enable_checkpointing = True,
    logger=tb_logger,
    callbacks=callbacks,
)

trainer.fit(model=model, train_dataloaders=dataloader)