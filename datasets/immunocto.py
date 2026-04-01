from pathlib import Path
from glob import glob
from skimage.io import imread
from numpy import expand_dims, clip
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class ImmunoctoDataset(Dataset):
    
    def __init__(
        self, 
        root_dir, 
        n_samples = 3
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
                
    def __len__(self):
        return len(self.idx_label)
        
    def __getitem__(self, idx):
        
        image   = imread(self.idx_image[idx])
        mask    = imread(self.idx_mask[idx])
        mask    = expand_dims(mask, axis = -1)
        # Clip to a boolean mask
        mask    = clip(mask, a_min = 0, a_max = 1)
        
        image   = image * mask
        # Must take shape of [N, C, W, H]
        # image   = image.reshape(3, 64, 64)
        label   = self.idx_label[idx]
        image   = transforms.ToTensor()(image)
        
        return image, label