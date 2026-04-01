import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ======================
# CONFIG
# ======================
SPLIT_DIR = Path("/project/aip-gregorys/yyx01056/data/tissue_vs_blood")
OUT_DIR = Path("/project/aip-gregorys/yyx01056/tissue_vs_blood/bbox")        # DIR to save extracted embeddings by applying bounding box masks for tissue dataset, and zero masks for blood dataset (since no masks available)
MODEL_PATH = "/project/aip-gregorys/yyx01056/dinobloom-s.pth"

BATCH_SIZE = 256  
NUM_WORKERS = 4

img_size = 224
eval_model = "dinov2_vits14"    ## DinoBloom-s (small) base model

embed_sizes = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536}
# ======================


def get_dino_bloom(modelpath=MODEL_PATH, modelname="dinov2_vits14"):
    # load the original DINOv2 model with the correct architecture and parameters.
    model = torch.hub.load('facebookresearch/dinov2', modelname)

    # load finetuned weights
    pretrained = torch.load(modelpath, map_location=torch.device('cpu'))

    # make correct state dict for loading
    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key or "ibot_head" in key:
            pass
        else:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value
   
    # corresponds to 224x224 image. patch size=14x14 => 16*16 patches
    pos_embed = nn.Parameter(torch.zeros(1, 257, embed_sizes[modelname]))
    model.pos_embed = pos_embed

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model


model = get_dino_bloom()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Resize to 224×224；
# Convert to tensor；
# Normalize with ImageNet mean/std as DINOv2 expects
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])


# ======================
# Dataset for efficient loading
# ======================
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform, mask_dir=None, use_mask=True):
        self.image_paths = image_paths
        self.transform = transform
        self.use_mask = use_mask
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")

            # Apply mask if available (CRC only)
            if self.use_mask and (self.mask_dir is not None):
                mask_path = self.mask_dir / image_path.name
                if mask_path.exists():
                    mask = Image.open(mask_path).convert("L")

                    image = image.resize((img_size, img_size))
                    mask = mask.resize((img_size, img_size))

                    black = Image.new("RGB", image.size, (0, 0, 0))
                    image = Image.composite(image, black, mask)
                else:
                    image = image.resize((img_size, img_size))
            else:
                image = image.resize((img_size, img_size))

            img_tensor = self.transform(image)
            ok = 1

        except Exception:
            # return a dummy tensor + ok=0, keep alignment
            img_tensor = torch.zeros(3, img_size, img_size, dtype=torch.float32)
            ok = 0

        return img_tensor, str(image_path), ok
 


def extract_split(split_name):

    # canonical order 
    lists_dir = SPLIT_DIR / "lists"
    names_file = lists_dir / f"{split_name}_out_names.txt"
    if names_file.exists():
        out_names = [x.strip() for x in names_file.read_text().splitlines() if x.strip()]
        images_dir = SPLIT_DIR / split_name / "images"
        img_paths = [images_dir / name for name in out_names]
    else:
        img_paths = sorted((SPLIT_DIR / split_name / "images").glob("*"))

    print(f"{split_name}: {len(img_paths)} images")

    if len(img_paths) == 0:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mask_dir = SPLIT_DIR / split_name / "masks"
    if not mask_dir.is_dir():
        mask_dir = None   # blood dataset all have no masks

    # Save path alignment file
    (OUT_DIR / f"paths_{split_name}.txt").write_text(
        "\n".join(map(str, img_paths)) + "\n"
    )

    embedding_dim = embed_sizes[eval_model]

    # Create disk-backed array to store embeddings efficiently. Use float16 to save space.
    emb_path = OUT_DIR / f"embeddings_{split_name}.float16.memmap"
    embeddings = np.memmap(
        emb_path,
        dtype=np.float16,
        mode="w+",
        shape=(len(img_paths), embedding_dim)
    )

    dataset = ImageDataset(img_paths, transform, mask_dir=mask_dir, use_mask=True)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    row = 0
    bad_paths = []

    with torch.no_grad():
        for imgs_tensor, path_batch, ok_batch in tqdm(dataloader, desc=split_name):

            imgs_tensor = imgs_tensor.to(device, non_blocking=True)
            ok_batch = ok_batch.numpy().astype(bool)

            if device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    features_dict = model.forward_features(imgs_tensor)
                    features = features_dict['x_norm_clstoken']
            else:
                features_dict = model.forward_features(imgs_tensor)
                features = features_dict['x_norm_clstoken']

            batch_embeddings = features.detach().cpu().to(torch.float16).numpy()

            if not ok_batch.all():
                for p, ok in zip(path_batch, ok_batch):
                    if not ok:
                        bad_paths.append(p)
                batch_embeddings[~ok_batch, :] = 0

            b = batch_embeddings.shape[0]
            embeddings[row:row+b] = batch_embeddings
            row += b

            if row % (BATCH_SIZE * 200) == 0:
                embeddings.flush()

    embeddings.flush()

    if bad_paths:
        (OUT_DIR / f"bad_images_{split_name}.txt").write_text("\n".join(bad_paths) + "\n")
        print(f"{split_name}: skipped/zeroed {len(bad_paths)} bad images (see bad_images_{split_name}.txt)")

    print(f"Saved {split_name} embeddings:", embeddings.shape)
    print("File:", emb_path)


for split in ["train", "val", "test"]:
    extract_split(split)