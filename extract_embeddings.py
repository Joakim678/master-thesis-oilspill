import time
from pathlib import Path
import numpy as np
import tifffile as tiff
import torch
from torchgeo.models import resnet50
from torchgeo.models.resnet import ResNet50_Weights


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DATA_ROOT = Path("root/to/project")
IMG_DIR = DATA_ROOT / "Dataset" / "images" / "Oil"
INDEX_DIR = DATA_ROOT / "patch_indices"
OUT_DIR = DATA_ROOT / "scene_embeddings"
OUT_DIR.mkdir(exist_ok=True)

PATCH = 128
GRID = 16
EMBED_DIM = 2048

SPLIT = "test"          # "train", "val", or "test"
MAX_SCENES = None         # Set to an integer to limit number of scenes processed (for quick testing)

WEIGHTS = ResNet50_Weights.SENTINEL1_ALL_MOCO


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clip01(x, lo, hi):
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo)


def load_split_as_scene_arrays(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)

    scene_ids = d["scene_ids"].astype(str)
    scene_idx = d["scene_idx"]
    patch_idx = d["patch_idx"]
    labels = d["label"].astype(np.float32)
    oil_frac = d["oil_frac"].astype(np.float32)

    num_scenes = len(scene_ids)
    labels_scene = np.zeros((num_scenes, GRID * GRID), dtype=np.float32)
    oil_scene = np.zeros((num_scenes, GRID * GRID), dtype=np.float32)

    labels_scene[scene_idx, patch_idx] = labels
    oil_scene[scene_idx, patch_idx] = oil_frac

    return scene_ids, labels_scene, oil_scene


def load_scene_as_patches(scene_id: str):
    img = tiff.imread(IMG_DIR / f"{scene_id}.tif")  # (2048,2048,2)

    patches = np.empty((GRID * GRID, 2, PATCH, PATCH), dtype=np.float32)

    k = 0
    for r in range(GRID):
        y0, y1 = r * PATCH, (r + 1) * PATCH
        for c in range(GRID):
            x0, x1 = c * PATCH, (c + 1) * PATCH

            p = img[y0:y1, x0:x1, :]
            vv = clip01(p[..., 0], -35.0, 5.0)
            vh = clip01(p[..., 1], -40.0, 0.0)

            patches[k, 0] = vv
            patches[k, 1] = vh
            k += 1

    return patches  # (256,2,128,128)


# --------------------------------------------------
# MODEL
# --------------------------------------------------
def build_frozen_resnet50():
    model = resnet50(weights=WEIGHTS)
    model.eval()
    return model


@torch.no_grad()
def extract_patch_embeddings(model, xb):
    captured = {}

    def hook(module, inputs, output):
        captured["feat"] = inputs[0].detach()

    handle = model.fc.register_forward_hook(hook)
    _ = model(xb)
    handle.remove()

    feats = captured["feat"].view(xb.size(0), -1)
    return feats  # (B,2048)


@torch.no_grad()
def embed_scene(model, scene_id: str, device: torch.device):
    patches = load_scene_as_patches(scene_id)   # (256,2,128,128)
    xb = torch.from_numpy(patches).to(device)

    feats = extract_patch_embeddings(model, xb)  # (256,2048)
    feats = feats.cpu().numpy()

    grid = feats.reshape(GRID, GRID, EMBED_DIM).transpose(2, 0, 1)  # (2048,16,16)
    return grid.astype(np.float32)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    device = choose_device()
    print("Device:", device)

    split_file = INDEX_DIR / f"patch_index_{SPLIT}.npz"
    scene_ids, labels_scene, oil_scene = load_split_as_scene_arrays(split_file)

    if MAX_SCENES is not None:
        scene_ids = scene_ids[:MAX_SCENES]
        labels_scene = labels_scene[:MAX_SCENES]
        oil_scene = oil_scene[:MAX_SCENES]

    split_out = OUT_DIR / SPLIT
    split_out.mkdir(parents=True, exist_ok=True)

    model = build_frozen_resnet50().to(device)

    overall_t0 = time.time()

    for i, scene_id in enumerate(scene_ids, start=1):
        t0 = time.time()

        x = embed_scene(model, scene_id, device)                      # (2048,16,16)
        y = labels_scene[i - 1].reshape(GRID, GRID).astype(np.float32)
        oil = oil_scene[i - 1].reshape(GRID, GRID).astype(np.float32)

        np.savez_compressed(
            split_out / f"{scene_id}.npz",
            embedding=x,
            label=y,
            oil_frac=oil,
            scene_id=scene_id,
        )

        dt = time.time() - t0
        avg = (time.time() - overall_t0) / i

        print(
            f"[{i:03d}/{len(scene_ids)}] scene={scene_id} "
            f"time={dt:.2f}s avg={avg:.2f}s/scene",
            flush=True,
        )

    total = time.time() - overall_t0
    print(f"\nFinished {len(scene_ids)} scenes in {total:.2f}s")
    print(f"Average: {total / len(scene_ids):.2f}s per scene")
    print("Saved to:", split_out)


if __name__ == "__main__":
    main()