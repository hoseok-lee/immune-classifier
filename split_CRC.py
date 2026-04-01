##### Split Immunocto CRC patients (CRC01–CRC30) into train/temp by PATIENT with an 8:2 ratio,
##### then split temp set into val/test 5:5, and create symlink folders.


from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path("/home/yyx01056/projects/aip-gregorys/yyx01056/data/CRC_Immunocto")
SEED = 42

patients = [f"CRC{i:02d}" for i in range(1, 31)]
missing = [p for p in patients if not (ROOT / p).is_dir()]
if missing:
    raise RuntimeError(f"Missing folders: {missing}")

print("Splitting CRC patients 80/10/10 (train/val/test)...")
# Split into 8:1:1 train/val/test
train, tmp = train_test_split(patients, test_size=0.2, random_state=SEED, shuffle=True)
val, test = train_test_split(tmp, test_size=0.5, random_state=SEED, shuffle=True)

print(f"Total patients: {len(patients)}")
print(f"Training patients: {len(train)}")
print(f"Validation patients: {len(val)}")
print(f"Test patients: {len(test)}")


# Write split lists
splits_dir = "/projects/aip-gregorys/yyx01056/data/CRC_Immunocto/splits_file"
splits_dir.mkdir(exist_ok=True)
(splits_dir / "train.txt").write_text("\n".join(sorted(train)) + "\n")
(splits_dir / "val.txt").write_text("\n".join(sorted(val)) + "\n")
(splits_dir / "test.txt").write_text("\n".join(sorted(test)) + "\n")
print("Wrote split txt files to:", splits_dir)

# Create symlink 
out_tree = "/projects/aip-gregorys/yyx01056/data/CRC_Immunocto/data_splits"
for split in ["train", "val", "test"]:
    (out_tree / split).mkdir(parents=True, exist_ok=True)

#create symlink if does not exist already
def link_patient(pid: str, split: str):
    src = ROOT / pid
    dst = out_tree / split / pid  # destination symlink path
    if dst.exists() or dst.is_symlink():
        return  # already there, skip
    dst.symlink_to(src)

for pid in train:
    link_patient(pid, "train")
for pid in val:
    link_patient(pid, "val")
for pid in test:
    link_patient(pid, "test")

print("Symlink folders created at:", out_tree)
print("Done.")

