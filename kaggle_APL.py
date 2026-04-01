from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


LABEL_TO_INT = {
    "immune": 1,
    "non_immune": 0,
}

def image_to_mask_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    idx = parts.index("slides")
    parts[idx] = "masks"
    return Path(*parts).with_suffix(".png")


def load_rgb(path: Path):
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path: Path):
    m = np.array(Image.open(path).convert("L"))
    return (m > 0).astype(np.uint8)


def get_bbox(mask, pad=2):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    h, w = mask.shape
    x1 = max(xs.min() - pad, 0)
    x2 = min(xs.max() + pad, w - 1)
    y1 = max(ys.min() - pad, 0)
    y2 = min(ys.max() + pad, h - 1)

    return x1, y1, x2, y2


class BloodCellCropDataset(Dataset):
    def __init__(
        self,
        csv_path,
        padding=4,
        image_size=224,
    ):
        self.df = pd.read_csv(csv_path)

        self.padding = padding

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # build mask paths
        self.df["mask_path"] = self.df["image_path"].apply(
            lambda x: str(image_to_mask_path(Path(x)))
        )

        # drop missing masks
        exists = self.df["mask_path"].apply(lambda x: Path(x).exists())
        self.df = self.df[exists].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        img = load_rgb(img_path)
        mask = load_mask(mask_path)

        bbox = get_bbox(mask, self.padding)

        if bbox is None:
            # fallback (rare)
            crop = img
        else:
            x1, y1, x2, y2 = bbox
            crop = img[y1:y2+1, x1:x2+1]

        crop = Image.fromarray(crop)
        crop = self.transform(crop)

        label = LABEL_TO_INT[row["label"]]

        return crop, label


def make_loader(csv_path, batch_size=32, shuffle=False, padding=4, image_size=224, num_workers=4):
    dataset = BloodCellCropDataset(
        csv_path=csv_path,
        padding=padding,
        image_size=image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataset, loader





if __name__ == "__main__":
    train_ds, train_loader = make_loader(
        "/home/yyx01056/scratch/splits/train.csv",
        batch_size=32,
        shuffle=True,
        padding=4,
        image_size=224,
        num_workers=4,
    )

    print(f"Dataset size: {len(train_ds)}")

    images, labels = next(iter(train_loader))

    print(images.shape)
    print(labels.shape)