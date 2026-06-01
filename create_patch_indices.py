from pathlib import Path
import json
import numpy as np
import tifffile as tiff

MSK_DIR = Path("Dataset/masks/Mask_oil")
SPLITS_PATH = Path("splits.json")
OUT_DIR = Path("patch_indices")
OUT_DIR.mkdir(exist_ok=True)

PATCH = 128
GRID = 16
TAU = 0.01  # 1% oil pixels in patch => positive

def compute_patch_labels(mask2d: np.ndarray, patch=128, tau=0.01):
    """Return (labels[256], oil_frac[256]) for a 2048x2048 binary mask."""
    m = (mask2d > 0).astype(np.uint8)
    H, W = m.shape
    assert H == 2048 and W == 2048, f"Unexpected mask shape {m.shape}"
    assert H % patch == 0 and W % patch == 0
    gh, gw = H // patch, W // patch
    assert gh == GRID and gw == GRID, f"Unexpected grid {(gh,gw)}"

    # (16,16,128,128)
    m4 = m.reshape(gh, patch, gw, patch).transpose(0, 2, 1, 3)
    frac = m4.mean(axis=(2, 3)).reshape(-1)  # 256
    y = (frac >= tau).astype(np.uint8)
    return y, frac.astype(np.float32)

def build_index(scene_ids, split_name):
    scene_ids = list(scene_ids)
    n = len(scene_ids)

    # We store: scene_idx (0..n-1), patch_idx (0..255), label (0/1), oil_frac
    scene_idx_all = []
    patch_idx_all = []
    label_all = []
    frac_all = []

    pos_per_scene = []

    for si, sid in enumerate(scene_ids):
        m = tiff.imread(MSK_DIR / f"{sid}.tif")
        y, frac = compute_patch_labels(m, patch=PATCH, tau=TAU)

        scene_idx_all.append(np.full(256, si, dtype=np.int32))
        patch_idx_all.append(np.arange(256, dtype=np.int16))
        label_all.append(y)
        frac_all.append(frac)

        pos_per_scene.append(int(y.sum()))

    scene_idx_all = np.concatenate(scene_idx_all)
    patch_idx_all = np.concatenate(patch_idx_all)
    label_all = np.concatenate(label_all)
    frac_all = np.concatenate(frac_all)

    # Save index
    out_path = OUT_DIR / f"patch_index_{split_name}.npz"
    np.savez_compressed(
        out_path,
        scene_ids=np.array(scene_ids),
        scene_idx=scene_idx_all,
        patch_idx=patch_idx_all,
        label=label_all,
        oil_frac=frac_all,
        patch=PATCH,
        tau=TAU,
        grid=GRID,
    )

    # Print stats
    total = label_all.size
    pos = int(label_all.sum())
    neg = total - pos
    print(f"\n[{split_name}] scenes: {n}")
    print(f"[{split_name}] patches: {total}  pos: {pos}  neg: {neg}  pos_rate: {pos/total:.4f}")
    print(f"[{split_name}] positives per scene: mean={np.mean(pos_per_scene):.2f}, "
          f"median={np.median(pos_per_scene):.1f}, min={np.min(pos_per_scene)}, max={np.max(pos_per_scene)}")
    print(f"Wrote: {out_path.resolve()}")

with open(SPLITS_PATH, "r") as f:
    splits = json.load(f)

build_index(splits["train"], "train")
build_index(splits["val"], "val")
build_index(splits["test"], "test")