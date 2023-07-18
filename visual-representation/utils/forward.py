import torch
import numpy as np

def get_features_labels(dataloader, model):
    feature_vectors = []
    labels = []

    with torch.no_grad():
        for images, target in dataloader:
            # images = images.to(device)
            features = model(images).flatten(start_dim=1)
            # features = model.backbone(images).flatten(start_dim=1).cpu()
            feature_vectors.append(features)
            labels.append(target.item())

    feature_vectors = np.concatenate(feature_vectors)
    return feature_vectors, labels