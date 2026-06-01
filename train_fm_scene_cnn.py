import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class PrecomputedSceneEmbeddingDataset(Dataset):
    def __init__(self, split_dir: str):
        self.split_dir = Path(split_dir)
        self.files = sorted(self.split_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.split_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i], allow_pickle=True)

        x = d["embedding"].astype(np.float32)   # (2048,16,16)
        y = d["label"].astype(np.float32)       # (16,16)
        oil = d["oil_frac"].astype(np.float32)  # (16,16)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(oil),
        )


class SmallSceneCNN(nn.Module):
    def __init__(self, in_channels: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)  # (B,1,16,16)
    
    


def compute_pos_weight(dataset: Dataset, device: torch.device) -> torch.Tensor:
    pos = 0.0
    total = 0

    for i in range(len(dataset)):
        _, y, _ = dataset[i]
        pos += float(y.sum().item())
        total += int(y.numel())

    neg = total - pos
    pos_weight = neg / max(pos, 1.0)

    print(f"train positives: {int(pos)} of {total}", flush=True)
    print(f"pos_weight: {pos_weight:.6f}", flush=True)

    return torch.tensor([pos_weight], dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, device, pos_weight, threshold=0.5, log_prefix="val"):
    model.eval()

    total_loss = 0.0
    n = 0

    tp = fp = tn = fn = 0
    num_pred_pos = 0
    num_true_pos = 0

    t0 = time.time()

    for step, (x, y, oil) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).unsqueeze(1)  # (B,1,16,16)

        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)

        prob = torch.sigmoid(logits)
        pred = (prob >= threshold).float()

        num_pred_pos += int((pred == 1).sum().item())
        num_true_pos += int((y == 1).sum().item())

        tp += int(((pred == 1) & (y == 1)).sum().item())
        fp += int(((pred == 1) & (y == 0)).sum().item())
        tn += int(((pred == 0) & (y == 0)).sum().item())
        fn += int(((pred == 0) & (y == 1)).sum().item())

        if step % 20 == 0:
            elapsed = time.time() - t0
            print(f"[{log_prefix}] step={step} elapsed={elapsed:.1f}s", flush=True)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    print(f"{log_prefix} predicted positives: {num_pred_pos}", flush=True)
    print(f"{log_prefix} true positives: {num_true_pos}", flush=True)

    return {
        "loss": total_loss / max(n, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc": acc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "predicted_positives": num_pred_pos,
        "true_positives": num_true_pos,
    }


def train_one_epoch(model, loader, optimizer, device, pos_weight, epoch, log_every=20):
    model.train()
    total_loss = 0.0
    n = 0

    t0 = time.time()

    for step, (x, y, oil) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).unsqueeze(1)  # (B,1,16,16)

        if step == 1:
            print(f"[train] epoch {epoch}: first batch input={tuple(x.shape)} target={tuple(y.shape)}", flush=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)

        if step % log_every == 0:
            elapsed = time.time() - t0
            avg = total_loss / max(n, 1)
            print(f"[train] epoch {epoch}: step={step} avg_loss={avg:.4f} elapsed={elapsed:.1f}s", flush=True)

    return total_loss / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, required=True)
    ap.add_argument("--val_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()

    print("Args:", vars(args), flush=True)

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"train_dir={train_dir}", flush=True)
    print(f"val_dir={val_dir}", flush=True)
    print(f"out_dir={out_dir}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    print("Building datasets...", flush=True)
    t0 = time.time()
    train_ds = PrecomputedSceneEmbeddingDataset(str(train_dir))
    val_ds = PrecomputedSceneEmbeddingDataset(str(val_dir))
    print(f"Datasets ready. train_scenes={len(train_ds)} val_scenes={len(val_ds)} | t={time.time()-t0:.2f}s", flush=True)

    pos_weight = compute_pos_weight(train_ds, device)

    print("Building dataloaders...", flush=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    print("Dataloaders ready", flush=True)

    print("Building model...", flush=True)
    model = SmallSceneCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("Model ready", flush=True)

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
            pos_weight,
            epoch=epoch,
            log_every=args.log_every,
        )

        val = evaluate(
            model,
            val_loader,
            device,
            pos_weight,
            threshold=args.threshold,
            log_prefix=f"val:e{epoch}",
        )

        dt = time.time() - epoch_t0

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_f1": val["f1"],
            "val_precision": val["precision"],
            "val_recall": val["recall"],
            "val_acc": val["acc"],
            "val_tp": val["tp"],
            "val_fp": val["fp"],
            "val_tn": val["tn"],
            "val_fn": val["fn"],
            "val_predicted_positives": val["predicted_positives"],
            "val_true_positives": val["true_positives"],
            "sec": dt,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_f1={val['f1']:.4f} "
            f"prec={val['precision']:.4f} "
            f"rec={val['recall']:.4f} "
            f"acc={val['acc']:.4f} | "
            f"val_loss={val['loss']:.4f} | "
            f"time={dt:.1f}s",
            flush=True,
        )

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "args": vars(args),
                },
                out_dir / "best.pt",
            )
            print(f"Saved best.pt (best_f1={best_f1:.4f})", flush=True)

        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            },
            out_dir / "last.pt",
        )

        (out_dir / "metrics.json").write_text(json.dumps(history, indent=2))
        print("Saved last.pt + metrics.json", flush=True)

    print("Done. Best val F1:", best_f1, flush=True)


if __name__ == "__main__":
    main()