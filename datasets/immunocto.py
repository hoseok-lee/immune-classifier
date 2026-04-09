from pathlib import Path
from glob import glob
from skimage.io import imread
from skimage.color import rgb2gray
from numpy import expand_dims, clip
from torch.utils.data import (
    Dataset, 
    DataLoader, 
    random_split, 
    WeightedRandomSampler
)
from torchvision import transforms
from typing import Optional
from PIL import Image
from collections import defaultdict

import numpy as np


class ImmunoctoDataset(Dataset):
    
    def __init__(
        self, 
        root_dir, 
        n_samples = 1e6,
        image_size = 224
    ):
        
        self.root_dir = Path(root_dir)
        
        # Get all sample/file names and organize by index
        # pytorch will use these indices to retrieve images at run-time

        self.patients = glob("*", root_dir = self.root_dir)
        self.n_samples = n_samples
        self.samples = []
        self.class_count = defaultdict(int)

        # Get all images and sample randomly
        # We do not want to bias based on patient/ordering
        for patient in self.patients:
            
            folder = self.root_dir / patient
            filenames = glob("*", root_dir = folder / "HE")
            
            for filename in filenames:
                
                label = int(filename.split("_")[0] != "other")
                self.class_count[label] += 1
                
                self.samples.append(
                    {
                        'image': folder / "HE" / filename,
                        'mask': folder / "mask" / filename,
                        # Class 0 if 'other', class 1 if immune cell
                        'label': label
                    }
                )
        
        # Random subset
        np.random.seed(0)
        self.samples = np.random.choice(
            self.samples,
            self.n_samples,
            replace = False
        )
                
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(360),
            transforms.ToTensor(),
        ])
                
    def __len__(self):
        return self.n_samples
        
    def __getitem__(self, idx):
        
        image   = imread(self.samples[idx]['image'])
        mask    = imread(self.samples[idx]['mask'])
        mask    = expand_dims(mask, axis = -1)
        # Clip to a boolean mask
        mask    = clip(mask, a_min = 0, a_max = 1)
        
        # Grayscale and apply mask
        # rgb2gray removes channels, add them back in
        image   = np.rint(rgb2gray(image) * 255).astype(np.uint8)
        image   = np.repeat(image[..., np.newaxis], 3, axis = -1)
        image   = image * mask
        
        label   = self.samples[idx]['label']
        image   = self.transform(Image.fromarray(image))
        
        return image, label
        
        
def get_dataloader(
    dataset,
    batch_size = 100,
    n_jobs = 1,
    subset = False
):
    
    if subset:
        return DataLoader(
            dataset,
            batch_size = batch_size,
            num_workers = n_jobs,
            persistent_workers = True,
            sampler = WeightedRandomSampler(
                weights = [
                    1. / dataset.dataset.class_count[
                        dataset.dataset.samples[idx]['label']
                    ]
                    for idx in dataset.indices
                ],
                num_samples = len(dataset),
                replacement = True,
            )
        )
    
    return DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = n_jobs,
        persistent_workers = True,
        sampler = WeightedRandomSampler(
            weights = [
                1. / dataset.class_count[sample['label']]
                for sample in dataset.samples
            ],
            num_samples = len(dataset),
            replacement = True,
        )
    )
        
        
def get_immunocto_loader(
    root_dir,
    n_samples = 1,
    splits: Optional[list] = [0.8, 0.1, 0.1],
    batch_size = 100,
    n_jobs = 1
):
    
    dataset = ImmunoctoDataset(
        root_dir = root_dir,
        n_samples = n_samples
    )
    
    # No split -> full dataset
    if splits is None:
        return get_dataloader(dataset)
    
    trainset, validset, testset = random_split(dataset, splits)
    
    trainloader = get_dataloader(trainset, subset = True)
    validloader = get_dataloader(validset, subset = True)
    testloader = get_dataloader(testset, subset = True)
    
    return trainloader, validloader, testloader
