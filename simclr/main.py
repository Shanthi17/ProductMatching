import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models, transforms, datasets
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
import numpy as np
import os
np.random.seed(0)

import warnings
warnings.filterwarnings('ignore')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='./dataset',
                    help='path to dataset')
parser.add_argument('--dataset-name', default='custom',
                    help='dataset name', choices=['custom', 'stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=os.cpu_count(), type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', default=False, action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

args = parser.parse_args()

class ContrastiveLearningViewGenerator():
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

print('Done Loading libraries')

# workers = os.cpu_count()
# data = './sub_dataset'
# args = arguments(workers, data)


if torch.cuda.is_available():
    args.device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True
else:
    args.device = torch.device('cpu')
    args.gpu_index = -1

print('Set cuda device')

color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomApply([color_jitter], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.ToTensor()])


train_dataset = datasets.ImageFolder(root=args.data,
                               transform=ContrastiveLearningViewGenerator(data_transforms, n_views=args.n_views))

print('Set dataset from folder')

train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.workers)

print('Created data loader')

model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,last_epoch=-1)

print('loaded model, optimizer, scheduler')

with torch.cuda.device(args.gpu_index):
    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    print('Started training')
    simclr.train(train_loader)
