from pathlib import Path
from glob import glob
from skimage.io import imread
from skimage.color import rgb2gray
from numpy import expand_dims, clip
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from typing import Optional
from PIL import Image

import numpy as np


class ImmunoctoDataset(Dataset):
    
    def __init__(
        self, 
        root_dir, 
        n_samples = 3,
        image_size = 224
    ):
        
        self.root_dir = Path(root_dir)
        
        # Get all sample/file names and organize by index
        # pytorch will use these indices to retrieve images at run-time

        self.samples = glob("*", root_dir = self.root_dir)
        self.n_samples = n_samples
        
        self.idx_image  = []
        self.idx_mask   = []
        self.idx_label  = []
        
        for sample in self.samples[:self.n_samples]:
            
            folder = self.root_dir / sample
            filenames = glob("*", root_dir = folder / "HE")
            
            for filename in filenames:
                self.idx_image.append(folder / "HE" / filename)
                self.idx_mask.append(folder / "mask" / filename)
                # Class 0 if 'other', class 1 if immune cell
                self.idx_label.append(int(filename.split("_")[0] != "other"))
                
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
                
    def __len__(self):
        return len(self.idx_label)
        
    def __getitem__(self, idx):
        
        image   = imread(self.idx_image[idx])
        mask    = imread(self.idx_mask[idx])
        mask    = expand_dims(mask, axis = -1)
        # Clip to a boolean mask
        mask    = clip(mask, a_min = 0, a_max = 1)
        
        # Grayscale and apply mask
        # rgb2gray removes channels, add them back in
        image   = np.rint(rgb2gray(image) * 255).astype(np.uint8)
        image   = np.repeat(image[..., np.newaxis], 3, axis = -1)
        image   = image * mask
        # Must take shape of [N, C, W, H]
        # image   = image.reshape(3, 64, 64)
        label   = self.idx_label[idx]
        image   = self.transform(Image.fromarray(image))
        
        return image, label
        
        
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
        return DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = n_jobs,
            persistent_workers = True
        )
    
    trainset, validset, testset = random_split(dataset, splits)
    
    trainloader = DataLoader(
        trainset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = n_jobs,
        persistent_workers = True
    )
    
    validloader = DataLoader(
        validset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = n_jobs,
        persistent_workers = True
    )
    
    testloader = DataLoader(
        testset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = n_jobs,
        persistent_workers = True
    )
    
    return trainloader, validloader, testloader
