import argparse
from pathlib import Path
import json
import time

import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchgeo.models import resnet50
from torchgeo.models.resnet import ResNet50_Weights


# -------------------------
# Dataset
# -------------------------
class SARSceneDataset(Dataset):
    """
    One dataset item = one scene.
    Returns:
      x: (256, 2, 128, 128)
      y: (256,)
      oil_frac: (256,)
    """

    def __init__(
        self,
        index_npz_path: str,
        img_dir: str,
        patch: int = 128,
        grid: int = 16,
        vv_clip=(-35.0, 5.0),
        vh_clip=(-40.0, 0.0),
        debug: bool = False,
    ):
        self.img_dir = Path(img_dir)
        self.patch = patch
        self.grid = grid
        self.vv_clip = vv_clip
        self.vh_clip = vh_clip
        self.debug = debug
        self._cache = {}
        self._cache_max = 2

        t0 = time.time()
        data = np.load(index_npz_path, allow_pickle=True)

        self.scene_ids = data["scene_ids"].astype(str)
        scene_idx = data["scene_idx"]
        patch_idx = data["patch_idx"]
        label = data["label"].astype(np.float32)
        oil_frac = data["oil_frac"].astype(np.float32)
        dt = time.time() - t0

        num_scenes = len(self.scene_ids)
        num_patches = self.grid * self.grid

        # Reshape labels and oil_frac into (num_scenes, 256)
        self.label_scene = np.zeros((num_scenes, num_patches), dtype=np.float32)
        self.oil_scene = np.zeros((num_scenes, num_patches), dtype=np.float32)

        self.label_scene[scene_idx, patch_idx] = label
        self.oil_scene[scene_idx, patch_idx] = oil_frac

        if self.debug:
            print(
                f"[scene-dataset] Loaded index: {index_npz_path} | "
                f"scenes={num_scenes} patches={len(label)} | t={dt:.2f}s",
                flush=True,
            )

    def __len__(self):
        return len(self.scene_ids)

    @staticmethod
    def _clip01(x, lo, hi):
        x = np.clip(x, lo, hi)
        return (x - lo) / (hi - lo)

    def _get_scene(self, sid: str):
        arr = self._cache.get(sid)
        if arr is not None:
            return arr

        path = self.img_dir / f"{sid}.tif"

        try:
            arr = tiff.memmap(path)
            if self.debug and len(self._cache) < 3:
                print(f"[scene-dataset] memmap opened: {path}", flush=True)
        except Exception:
            arr = tiff.imread(path)
            if self.debug and len(self._cache) < 3:
                print(f"[scene-dataset] imread fallback: {path}", flush=True)

        self._cache[sid] = arr
        if len(self._cache) > self._cache_max:
            self._cache.pop(next(iter(self._cache)))

        return arr

    def __getitem__(self, i):
        sid = self.scene_ids[i]
        img = self._get_scene(sid)  # (2048, 2048, 2)

        patches = np.empty((self.grid * self.grid, 2, self.patch, self.patch), dtype=np.float32)

        k = 0
        for r in range(self.grid):
            y0, y1 = r * self.patch, (r + 1) * self.patch
            for c in range(self.grid):
                x0, x1 = c * self.patch, (c + 1) * self.patch
                p = img[y0:y1, x0:x1, :]

                vv = self._clip01(p[..., 0], *self.vv_clip)
                vh = self._clip01(p[..., 1], *self.vh_clip)

                patches[k, 0] = vv
                patches[k, 1] = vh
                k += 1

        y = self.label_scene[i]      # (256,)
        oil = self.oil_scene[i]      # (256,)

        return (
            torch.from_numpy(patches),                 # (256, 2, 128, 128)
            torch.from_numpy(y),                       # (256,)
            torch.from_numpy(oil),                     # (256,)
        )

# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, log_prefix="val"):
    model.eval()
    tp = fp = tn = fn = 0
    total_loss = 0.0
    n = 0

    all_true = []
    all_pred = []
    all_prob = []
    all_oil = []

    t0 = time.time()
    for step, (x, y, oil_frac) in enumerate(loader, start=1):
        B = x.size(0)

        x = x.to(device, non_blocking=True).view(B * 256, 2, 128, 128)
        y = y.to(device, non_blocking=True).view(B * 256, 1)
        oil_frac = oil_frac.view(B * 256)

        logits = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)

        probs = torch.sigmoid(logits)
        pred = (probs >= threshold).float()

        tp += int(((pred == 1) & (y == 1)).sum().item())
        fp += int(((pred == 1) & (y == 0)).sum().item())
        tn += int(((pred == 0) & (y == 0)).sum().item())
        fn += int(((pred == 0) & (y == 1)).sum().item())

        all_true.append(y.view(-1).cpu().numpy())
        all_pred.append(pred.view(-1).cpu().numpy())
        all_prob.append(probs.view(-1).cpu().numpy())
        all_oil.append(oil_frac.view(-1).cpu().numpy())

        if step % 20 == 0:
            elapsed = time.time() - t0
            print(f"[{log_prefix}] step={step} elapsed={elapsed:.1f}s", flush=True)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return {
        "loss": total_loss / max(n, 1),
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "y_true": np.concatenate(all_true),
        "y_pred": np.concatenate(all_pred),
        "y_prob": np.concatenate(all_prob),
        "oil_frac": np.concatenate(all_oil),
    }

def train_one_epoch(model, loader, optimizer, device, epoch: int, log_every: int = 20):
    model.train()
    total_loss = 0.0
    n = 0

    t0 = time.time()
    for step, (x, y, _) in enumerate(loader, start=1):
        # x: (B, 256, 2, 128, 128)
        # y: (B, 256)
        B = x.size(0)

        x = x.to(device, non_blocking=True).view(B * 256, 2, 128, 128)
        y = y.to(device, non_blocking=True).view(B * 256, 1)

        if step == 1:
            print(f"[train] first batch: scenes={B} -> patches={B*256} x={tuple(x.shape)}", flush=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)

        if step % log_every == 0:
            elapsed = time.time() - t0
            avg = total_loss / max(n, 1)
            print(f"[train] step={step} avg_loss={avg:.4f} elapsed={elapsed:.1f}s", flush=True)

    return total_loss / max(n, 1)


def metrics_by_oil_fraction(y_true, y_pred, oil_frac, bins):
    rows = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        # include right edge only for last bin
        if hi == bins[-1]:
            mask = (oil_frac >= lo) & (oil_frac <= hi)
            bin_name = f"[{lo:.2f}, {hi:.2f}]"
        else:
            mask = (oil_frac >= lo) & (oil_frac < hi)
            bin_name = f"[{lo:.2f}, {hi:.2f})"

        n = int(mask.sum())
        if n == 0:
            rows.append({
                "bin": bin_name,
                "count": 0,
                "acc": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
            })
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())

        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        rows.append({
            "bin": bin_name,
            "count": n,
            "acc": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        })

    return rows


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Root that contains Dataset/ and patch_indices/")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--log_every", type=int, default=200, help="Print train progress every N steps")
    ap.add_argument("--debug_dataset", action="store_true", help="Verbose dataset memmap logs")
    args = ap.parse_args()

    print("Args:", vars(args), flush=True)

    data_root = Path(args.data_root)
    img_dir = data_root / "Dataset/images/Oil"
    idx_dir = data_root / "patch_indices"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Resolved paths:", flush=True)
    print(f"  img_dir={img_dir}", flush=True)
    print(f"  idx_dir={idx_dir}", flush=True)
    print(f"  out_dir={out_dir}", flush=True)

    # Basic file sanity checks
    for p in [img_dir, idx_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    # Datasets/loaders
    print("Building datasets...", flush=True)
    t0 = time.time()
    train_ds = SARSceneDataset(str(idx_dir / "patch_index_train.npz"), str(img_dir), debug=args.debug_dataset)
    val_ds = SARSceneDataset(str(idx_dir / "patch_index_val.npz"), str(img_dir), debug=args.debug_dataset)
    print(
        f"Datasets ready. train_scenes={len(train_ds)} val_scenes={len(val_ds)} | t={time.time()-t0:.2f}s",
        flush=True,
    )

    print("Building dataloaders...", flush=True)
    t0 = time.time()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    print(f"Dataloaders ready | t={time.time()-t0:.2f}s", flush=True)

    # Model
    print("Building model (TorchGeo resnet50 + Sentinel-1 MoCo weights)...", flush=True)
    t0 = time.time()
    weights = ResNet50_Weights.SENTINEL1_ALL_MOCO
    model = resnet50(weights=weights)

    # Replace classifier head -> 1 logit
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
        print(f"Replaced model.fc head (in_features={in_features})", flush=True)
    elif hasattr(model, "head") and isinstance(model.head, nn.Module):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, 1)
        print(f"Replaced model.head head (in_features={in_features})", flush=True)
    else:
        raise RuntimeError("Could not find classifier head (fc/head) to replace.")

    model = model.to(device)
    print(f"Model ready on {device} | t={time.time()-t0:.2f}s", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("Optimizer: AdamW", flush=True)

    history = []
    best_f1 = -1.0

    print(f"Starting training for {args.epochs} epochs...", flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            log_every=args.log_every,
        )

        val = evaluate(model, val_loader, device, threshold=args.threshold, log_prefix=f"val:e{epoch}")

        bins = np.linspace(0.0, 1.0, 11)  # 0-10%, 10-20%, ..., 90-100%
        bin_rows = metrics_by_oil_fraction(
            val["y_true"],
            val["y_pred"],
            val["oil_frac"],
            bins,
        )



        dt = time.time() - epoch_t0

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{
                f"val_{k}": v
                for k, v in val.items()
                if k not in ["y_true", "y_pred", "y_prob", "oil_frac"]
            },
            "oil_fraction_bins": bin_rows,
            "sec": dt,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_f1={val['f1']:.4f} prec={val['precision']:.4f} rec={val['recall']:.4f} "
            f"acc={val['acc']:.4f} | "
            f"val_loss={val['loss']:.4f} | "
            f"time={dt:.1f}s",
            flush=True,
        )

        print("Validation metrics by oil fraction:", flush=True)
        for r in bin_rows:
            print(
                f"  {r['bin']}: "
                f"count={r['count']} "
                f"acc={r['acc']:.4f} "
                f"prec={r['precision']:.4f} "
                f"rec={r['recall']:.4f} "
                f"f1={r['f1']:.4f}",
                flush=True,
            )




        # Save best
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_f1": best_f1}, out_dir / "best.pt")
            print(f"Saved best.pt (best_f1={best_f1:.4f})", flush=True)

        # Save latest every epoch
        torch.save({"model": model.state_dict(), "epoch": epoch}, out_dir / "last.pt")
        (out_dir / "metrics.json").write_text(json.dumps(history, indent=2))
        print("Saved last.pt + metrics.json", flush=True)

    print("Done. Best val F1:", best_f1, flush=True)


if __name__ == "__main__":
    main()