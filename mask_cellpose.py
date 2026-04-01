from pathlib import Path
import numpy as np
from tqdm import tqdm
from cellpose import models, io
import torch

print("torch.cuda.is_available():", torch.cuda.is_available())

# =============================
# CONFIG
# =============================
input_dir = Path("/home/yyx01056/scratch/archive/All")

flow_threshold = 0.4
cellprob_threshold = 0.0
tile_norm_blocksize = 0

# Cellpose-SAM
model = models.CellposeModel(
    gpu=torch.cuda.is_available(),
    pretrained_model="cpsam"
)

# =============================
# HELPERS
# =============================
def get_largest_instance_mask(masks: np.ndarray):
    """
    Cellpose instance mask might produce multiple cell masks.
    Keep only the largest mask for the cell of interest.
    Return binary mask: {0,1}
    """
    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids != 0]

    if len(cell_ids) == 0:
        return None

    best_id = None
    best_area = -1

    for cid in cell_ids:
        area = np.sum(masks == cid)
        if area > best_area:
            best_area = area
            best_id = cid

    return (masks == best_id).astype(np.uint8)


def get_mask_path(img_path: Path):
    """
    Convert image path from slides to masks:
    Patient_xx/slides/class_name/img.jpg
    -->
    Patient_xx/masks/class_name/img.png
    """
    parts = list(img_path.parts)
    try:
        idx = parts.index("slides")
    except ValueError:
        raise ValueError(f"'slides' not found in path: {img_path}")

    parts[idx] = "masks"
    out = Path(*parts)
    return out.with_suffix(".png")


# =============================
# MAIN
# =============================
files = sorted(input_dir.glob("Patient_*/slides/*/*.jpg"))
print(f"Total images found: {len(files)}")

n_success = 0
n_failed = 0
n_empty = 0

for f in tqdm(files):
    try:
        img = io.imread(f)

        masks, flows, styles = model.eval(
            img,
            batch_size=1,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            normalize={"tile_norm_blocksize": tile_norm_blocksize},
        )

        single_mask = get_largest_instance_mask(masks)
        if single_mask is None:
            n_empty += 1
            continue

        mask_path = get_mask_path(f)
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        # save as 0/255, uint8 png
        io.imsave(
            mask_path,
            (single_mask * 255).astype(np.uint8),
        )
        n_success += 1

    except Exception as e:
        n_failed += 1
        print(f"[ERROR] {f}: {e}")

print("\nDone.")
print(f"Saved masks: {n_success}")
print(f"No mask found: {n_empty}")
print(f"Failed: {n_failed}")