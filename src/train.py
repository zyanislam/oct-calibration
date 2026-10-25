"""Training and evaluation for the Step 5 baseline.

set_seed            fix random, numpy and torch
class_weights       loss weights so every class pushes the model about equally
cosine_with_warmup  linear warm-up, then cosine decay to zero (per step)
first_loss          debugging check 2: should be about ln(4) = 1.39 before training
overfit_one_batch   debugging check 3: the loss on one fixed batch should approach 0
train_one_epoch     one pass over train with mixed precision
evaluate            accuracy, macro-F1 and the raw logits, labels and row indices
fit                 the full run: log per epoch, keep best.pt by val macro-F1, save logits
"""
from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(labels: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """n_total / (num_classes * n_class): rare classes get larger weights, mean weight per scan is 1."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    return counts.sum() / (num_classes * counts)


def cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    """Learning rate rises linearly for warmup_steps, then follows a cosine down to 0."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def subset_loader(loader: DataLoader, n: int, seed: int = 42) -> DataLoader:
    """A shuffled loader over n random scans of loader's dataset (for the smoke test)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(loader.dataset), generator=g)[:n].tolist()
    return DataLoader(Subset(loader.dataset, idx), batch_size=loader.batch_size, shuffle=True,
                      num_workers=loader.num_workers, pin_memory=loader.pin_memory)


def _autocast(device: torch.device):
    # Mixed precision on GPU only; on CPU it would be slower, so it is switched off there.
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


@torch.no_grad()
def first_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Unweighted cross-entropy on one batch before any training."""
    model.eval()
    images, labels, _ = next(iter(loader))
    with _autocast(device):
        logits = model(images.to(device))
    return nn.functional.cross_entropy(logits.float(), labels.to(device)).item()


def overfit_one_batch(model: nn.Module, loader: DataLoader, device: torch.device,
                      steps: int = 50, lr: float = 1e-3) -> list[float]:
    """Train on the same batch repeatedly. A working loop drives the loss towards 0."""
    model.train()
    images, labels, _ = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            loss = nn.functional.cross_entropy(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    return losses


def train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn, device) -> float:
    model.train()
    total, n = 0.0, 0
    for images, labels, _ in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            loss = loss_fn(model(images), labels)
        scaler.scale(loss).backward()
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # If the scaler found inf gradients it skipped this optimizer step (it lowers the
        # scale); skip the scheduler too, so the learning-rate schedule stays in step.
        if scaler.get_scale() >= scale:
            scheduler.step()
        total += loss.item() * len(labels)
        n += len(labels)
    return total / n


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Metrics plus raw logits (before softmax), labels and dataset row indices."""
    model.eval()
    logits, labels, rows = [], [], []
    for images, y, idx in loader:
        with _autocast(device):
            out = model(images.to(device, non_blocking=True))
        logits.append(out.float().cpu())
        labels.append(y)
        rows.append(torch.as_tensor(idx))
    logits, labels, rows = torch.cat(logits), torch.cat(labels), torch.cat(rows)
    pred = logits.argmax(1)
    return {
        "loss": nn.functional.cross_entropy(logits, labels).item(),   # unweighted, so epochs compare
        "accuracy": (pred == labels).float().mean().item(),
        "macro_f1": f1_score(labels, pred, average="macro"),
        "logits": logits, "labels": labels, "rows": rows,
    }


def fit(model: nn.Module, loaders: dict, out_dir, epochs: int = 8, lr: float = 3e-4,
        weight_decay: float = 1e-4, warmup_fraction: float = 0.05,
        weights: torch.Tensor | None = None, device: torch.device | None = None) -> list[dict]:
    """Train, keep best.pt by val macro-F1, then save val and test logits from best.pt.

    Writes to out_dir: best.pt, log.csv, logits_val.pt, logits_test.pt.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(loaders["train"])
    scheduler = cosine_with_warmup(optimizer, total_steps, warmup_steps=max(1, int(warmup_fraction * total_steps)))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    log, best = [], -1.0
    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss = train_one_epoch(model, loaders["train"], optimizer, scheduler, scaler, loss_fn, device)
        val = evaluate(model, loaders["val"], device)
        row = {"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val["loss"], 4),
               "val_accuracy": round(val["accuracy"], 4), "val_macro_f1": round(val["macro_f1"], 4),
               "lr_end": f"{scheduler.get_last_lr()[0]:.2e}", "seconds": round(time.time() - t)}
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            torch.save(model.state_dict(), out_dir / "best.pt")
            row["saved"] = "best.pt"
        log.append(row)
        print(row)
        with open(out_dir / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0].keys()) + ["saved"], extrasaction="ignore")
            w.writeheader()
            w.writerows(log)

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    for split in ("val", "test"):
        r = evaluate(model, loaders[split], device)
        torch.save({k: r[k] for k in ("logits", "labels", "rows")}, out_dir / f"logits_{split}.pt")
        print(f"best.pt on {split}: accuracy {r['accuracy']:.4f}, macro-F1 {r['macro_f1']:.4f}")
    return log
