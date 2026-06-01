from pathlib import Path
import json
import random


IMG_DIR = Path("Dataset/images/Oil")
OUT_PATH = Path("splits.json")

SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# Collect IDs (filestems like "00000")
ids = sorted([p.stem for p in IMG_DIR.glob("*.tif")])
n = len(ids)
assert n == 1200, f"Expected 1200 images, got {n}"

# Shuffle deterministically
rng = random.Random(SEED)
rng.shuffle(ids)

n_train = int(round(TRAIN_FRAC * n))
n_val = int(round(VAL_FRAC * n))
# ensure exact total
n_test = n - n_train - n_val

train_ids = ids[:n_train]
val_ids = ids[n_train:n_train + n_val]
test_ids = ids[n_train + n_val:]

print("Counts:", len(train_ids), len(val_ids), len(test_ids))
print("Example train IDs:", train_ids[:5])
print("Example val IDs  :", val_ids[:5])
print("Example test IDs :", test_ids[:5])

splits = {
    "seed": SEED,
    "fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
    "counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
    "train": train_ids,
    "val": val_ids,
    "test": test_ids,
}

OUT_PATH.write_text(json.dumps(splits, indent=2))
print("Wrote:", OUT_PATH.resolve())

# Optional: also write txt lists
Path("train_ids.txt").write_text("\n".join(train_ids) + "\n")
Path("val_ids.txt").write_text("\n".join(val_ids) + "\n")
Path("test_ids.txt").write_text("\n".join(test_ids) + "\n")
print("Also wrote train_ids.txt / val_ids.txt / test_ids.txt")