import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from models.resnet_simclr import ResNetSimCLR
import matplotlib.cm as cm
import argparse
import os
from utils import forward, load_dataset

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='./sub_dataset', help='path to dataset')
# parser.add_argument('--checkpoint', help='trained model checkpoint', required=True)
parser.add_argument('--filename', help='pca filename', required=True)
args = parser.parse_args()

# Data Loading
dataset, dataloader, idx_to_class = load_dataset.dataset(args.data)


# Model Loading
# model = ResNetSimCLR(base_model='resnet18', out_dim=128).to(device)
# checkpoint = torch.load(args.checkpoint, map_location=device)
# state_dict = checkpoint['state_dict']  
# log = model.load_state_dict(state_dict, strict=False)

# model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
model = torch.hub.load('facebookresearch/swav:main', 'resnet50')

# Extract feature vectors
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

save_path = os.path.join('results/pca', f"{args.filename}.png")
plt.savefig(save_path)
plt.close()
