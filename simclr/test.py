import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt 
import torch.backends.cudnn as cudnn
from torchvision import models, transforms, datasets
from models.resnet_simclr import ResNetSimCLR
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='./sub_dataset', help='path to dataset')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=os.cpu_count(), type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--checkpoint', help='trained model checkpoint', required=True)
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')

args = parser.parse_args()

if torch.cuda.is_available():
    args.device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True
else:
    args.device = torch.device('cpu')
    args.gpu_index = -1


class ContrastiveLearningViewGenerator():
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
data_transforms = transforms.Compose([transforms.Resize(size=(224,224)),
                                        transforms.ToTensor()])

class ImageFolderWithFilenames(datasets.ImageFolder):
  def __getitem__(self, index):
    img, target = super().__getitem__(index)
    filename = self.imgs[index][0]
    return img, target, filename

dataset = ImageFolderWithFilenames(root=args.data,
                               transform=ContrastiveLearningViewGenerator(data_transforms, n_views=1))

# dataset = ImageFolderWithFilenames(root=args.data)

dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)    

# tc = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor()              
#     ])

# dataset = ImageFolderWithFilenames(args.data, transform=tc)
# dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=200, shuffle=False)

model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim).to(args.device)
checkpoint = torch.load(args.checkpoint, map_location=args.device)
state_dict = checkpoint['state_dict']  

log = model.load_state_dict(state_dict, strict=False)

# model = torch.hub.load('facebookresearch/swav:main', 'resnet50')

def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, _, fnames in dataloader:
            img = img[0]
            img = img.to(args.device)
            emb = model.backbone(img).flatten(start_dim=1).cpu()
            # emb = model(img)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames


model.eval()
embeddings, filenames = generate_embeddings(model, dataset_loader)

np.save('embeddings_npy/embed_full.npy', np.array(embeddings, dtype=object), allow_pickle=True)
np.save('filenames.npy', np.array(filenames, dtype=object), allow_pickle=True )
embeddings = np.load('embeddings_npy/embed_full.npy', allow_pickle=True)

def get_image_as_np_array(filename: str):
    """Returns an image as an numpy array"""
    img = Image.open(filename)
    return np.asarray(img)


def plot_knn_examples(embeddings, filenames, n_neighbors=5, num_examples=20):
    """Plots multiple rows of random images with their nearest neighbors"""
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute', n_jobs=-1).fit(embeddings)
    # nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # get 5 random samples
    # samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)
    samples_idx = [20843, 39678, 36737,  1436, 26746, 40869, 53771, 518, 46944, 33751, 10303, 15298, 42185, 32155, 6393, 1549, 20593, 52356, 5669, 24102]

    # loop through our randomly picked samples
    for idx in samples_idx:
        fig = plt.figure()
        # loop through their nearest neighbors
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            # add the subplot
            ax = fig.add_subplot(1, len(indices[idx]), plot_x_offset + 1)
            # get the correponding filename for the current index
            fname = os.path.join(filenames[neighbor_idx])
            # plot the image
            plt.imshow(get_image_as_np_array(fname))
            # set the title to the distance of the neighbor
            ax.set_title(f"d={distances[idx][plot_x_offset]:.3f}")
            # let's disable the axis
            plt.axis("off")

        save_path = os.path.join('new_results/simclr_1.0_cosine', f"figure_{idx}.png")
        plt.savefig(save_path)
        plt.close()

plot_knn_examples(embeddings, filenames)
