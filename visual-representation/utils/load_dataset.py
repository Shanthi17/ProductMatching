from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch

def dataset(data_path):
    ## Generic transform to apply on images - resize to 224, 244 and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load the dataset
    dataset = datasets.ImageFolder(data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Get class to index dictionary - later used for plot in PCA and Reverse the class-to-index mapping to get index-to-class mapping
    class_to_idx = dataset.class_to_idx
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    return dataset, dataloader, idx_to_class