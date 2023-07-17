# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.transforms.swav_transform import SwaVTransform
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

class SwaV(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        # Freeze the weights
        for param in resnet.parameters():
            param.requires_grad = False
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SwaVProjectionHead(512, 512, 128)
        self.prototypes = SwaVPrototypes(128, n_prototypes=512)
        self.criterion = SwaVLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p

    def training_step(self, batch, batch_idx):
        self.prototypes.normalize()
        views = batch[0]
        multi_crop_features = [self.forward(view.to(self.device)) for view in views]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = self.criterion(high_resolution, low_resolution)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim


model = SwaV()

transform = SwaVTransform()
# we ignore object detection annotations by setting target_transform to return 0
# dataset = torchvision.datasets.VOCDetection(
#     "datasets/pascal_voc",
#     download=True,
#     transform=transform,
#     target_transform=lambda t: 0,
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