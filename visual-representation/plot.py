import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import os
from util import forward, load_dataset

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='./sub_dataset', help='path to dataset')
# parser.add_argument('--checkpoint', help='trained model checkpoint', required=True)
parser.add_argument('--filename', help='filename', required=True)
parser.add_argument('--representation',metavar='pca/tsne', help='type of visualization', required=True)
args = parser.parse_args()

# Data Loading
dataset, dataloader, idx_to_class = load_dataset.dataset(args.data)

# Model Loading
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
# model = torch.hub.load('facebookresearch/swav:main', 'resnet50')

# Extract feature vectors
feature_vectors, labels = forward.get_features_labels(dataloader, model)

# Load PCA with 2 components and fit the feature vectors
if args.representation == 'pca':
    rep = PCA(n_components=2)
    rep_result = rep.fit_transform(feature_vectors)
elif args.representation == 'tsne':
    rep = TSNE(n_components=2, random_state=42)
    rep_result = rep.fit_transform(feature_vectors)

# Generate a color map with 1000 distinct colors
color_map = cm.get_cmap('tab20', len(dataset.classes))

plt.figure(figsize=(10, 8))

for i in range(len(rep_result)):
    label = labels[i]
    color = color_map(label)
    plt.scatter(rep_result[i, 0], rep_result[i, 1], color=color, alpha=0.7)

# Create a legend for the color-coded labels
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map(label), label=idx_to_class[label])
                   for label in list(set(labels))]
plt.legend(handles=legend_elements, title='Class Labels', loc='upper right')

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title(f'2D {args.representation} Map of Image Feature Vectors')

save_path = os.path.join(f'results/{args.representation}', f"{args.filename}.png")
plt.savefig(save_path)
plt.close()
