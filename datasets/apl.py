from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageFile

from skimage.color import rgb2gray
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def image_to_mask_path(img_path: Path):
    
    parts = list(img_path.parts)
    idx = parts.index("slides")
    parts[idx] = "masks"
    
    return Path(*parts).with_suffix(".png")


class APLDataset(Dataset):
    
    def __init__(
        self, 
        csv_path, 
        image_size = 224
    ):
        
        self.df = pd.read_csv(csv_path)

        self.df["image_path"] = self.df["image_path"].apply(lambda x: str(Path(x).resolve()))
        self.df["mask_path"] = self.df["image_path"].apply(
            lambda x: str(image_to_mask_path(Path(x)))
        )

        image_exists = self.df["image_path"].apply(lambda x: Path(x).exists())
        mask_exists = self.df["mask_path"].apply(lambda x: Path(x).exists())
        keep = image_exists & mask_exists

        # print(f"Loaded rows: {len(self.df)}")
        # print(f"Existing images: {image_exists.sum()} / {len(self.df)}")
        # print(f"Existing masks:  {mask_exists.sum()} / {len(self.df)}")
        # print(f"Keeping rows:    {keep.sum()} / {len(self.df)}")

        self.df = self.df[keep].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No valid samples found in {csv_path}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(360),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]

        image   = np.array(Image.open(row["image_path"]).convert("RGB"))
        mask    = np.array(Image.open(row["mask_path"]).convert("L"))
        mask    = (mask > 0).astype(np.uint8)[..., np.newaxis]

        # Grayscale and apply mask
        # rgb2gray removes channels, add them back in
        image   = np.rint(rgb2gray(image) * 255).astype(np.uint8)
        image   = np.repeat(image[..., np.newaxis], 3, axis = -1)
        image   = image * mask

        label   = int(row["label"])
        image   = self.transform(Image.fromarray(image))

        return image, label