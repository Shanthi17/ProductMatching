import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from models.resnet_simclr import ResNetSimCLR
import numpy as np
import matplotlib.cm as cm
import argparse
import os
from utils import forward

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='./sub_dataset', help='path to dataset')
parser.add_argument('--checkpoint', help='trained model checkpoint', required=True)
parser.add_argument('--filename', help='pca filename', required=True)
args = parser.parse_args()

## Generic transform to apply on images - resize to 224, 244 and convert to tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the dataset
dataset = datasets.ImageFolder(args.data, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# selected_folder_names = [d for d in os.listdir(args.data) if len(os.listdir(os.path.join(args.data, d)))>=5]

# subset_indices = []
# for idx, (image, label) in enumerate(dataset):
#     if dataset.classes[label] in selected_folder_names:
#         subset_indices.append(idx)

# subset_dataset = Subset(dataset, subset_indices)


# Get class to index dictionary - later used for plot in PCA and Reverse the class-to-index mapping to get index-to-class mapping
class_to_idx = dataset.class_to_idx
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

# model = ResNetSimCLR(base_model='resnet18', out_dim=128).to(device)
# checkpoint = torch.load(args.checkpoint, map_location=device)
# state_dict = checkpoint['state_dict']  
# log = model.load_state_dict(state_dict, strict=False)

# model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
model = torch.hub.load('facebookresearch/swav:main', 'resnet50')

feature_vectors, labels = forward.get_features_labels(dataloader, model)

# Load PCA with 2 components and fit the feature vectors
pca = PCA(n_components=2)
pca_result = pca.fit_transform(feature_vectors)

# Generate a color map with 1000 distinct colors
color_map = cm.get_cmap('tab20', len(dataset.classes))

plt.figure(figsize=(10, 8))

for i in range(len(pca_result)):
    label = labels[i]
    color = color_map(label)
    plt.scatter(pca_result[i, 0], pca_result[i, 1], color=color, alpha=0.7)

# Create a legend for the color-coded labels
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map(label), label=idx_to_class[label])
                   for label in list(set(labels))]
plt.legend(handles=legend_elements, title='Class Labels', loc='upper right')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA Map of Image Feature Vectors')

save_path = os.path.join('pca', f"{args.filename}.png")
plt.savefig(save_path)
plt.close()
