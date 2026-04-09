import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import APL_SPLIT_PATH, APL_PATH

# ----------------------------
# CONFIG
# ----------------------------
ROOT_DIR = Path(APL_PATH)
OUTPUT_DIR = Path(APL_SPLIT_PATH)
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

RANDOM_SEED = 42

# binary label mapping (immune vs non-immune)
LABEL_MAP = {
    "eosinophil": "immune",
    "lymphocyte": "immune",
    "lymphocyte_variant": "immune",
    "segmented_neutrophil": "immune",
    "band_neutrophil": "immune",
    "smudge_cell": "immune",
    "erythroblast": "non_immune",
    "giant_thrombocyte": "non_immune",
}

BINARY_TO_INT = {
    "immune": 1,
    "non_immune": 0,
}


def get_all_patients(root_dir: Path):
    patients = [
        p.name for p in root_dir.iterdir()
        if p.is_dir() and p.name.startswith("Patient")
    ]
    return sorted(patients)


def split_patients(patients):
    train_patients, test_patients = train_test_split(
        patients,
        test_size=0.15,
        random_state=RANDOM_SEED
    )

    train_patients, val_patients = train_test_split(
        train_patients,
        test_size=0.1765,  # ~15% overall
        random_state=RANDOM_SEED
    )

    return train_patients, val_patients, test_patients


def collect_image_metadata(root_dir: Path, patient_list, split_name: str):
    rows = []

    for patient_id in patient_list:
        patient_dir = root_dir / patient_id / "slides"
        if not patient_dir.exists():
            continue

        for cell_type_dir in patient_dir.iterdir():
            if not cell_type_dir.is_dir():
                continue

            cell_type = cell_type_dir.name

            if cell_type not in LABEL_MAP:
                continue

            binary_label = LABEL_MAP[cell_type]
            label_int = BINARY_TO_INT[binary_label]

            for img_path in cell_type_dir.glob("*.jpg"):
                rows.append({
                    "patient_id": patient_id,
                    "image_path": str(img_path.resolve()),  # absolute path
                    "cell_type": cell_type,
                    "binary_label": binary_label,
                    "label": label_int,
                    "split": split_name,
                })

    return pd.DataFrame(rows)


def main():
    patients = get_all_patients(ROOT_DIR)
    print(f"Total patients: {len(patients)}")

    train_p, val_p, test_p = split_patients(patients)

    train_df = collect_image_metadata(ROOT_DIR, train_p, "train")
    val_df = collect_image_metadata(ROOT_DIR, val_p, "val")
    test_df = collect_image_metadata(ROOT_DIR, test_p, "test")

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df.to_csv(OUTPUT_DIR / "all_splits.csv", index=False)

    print("CSV split files saved.")
    print(full_df["split"].value_counts())
    print(full_df.head())


if __name__ == "__main__":
    main()