"""Step 7 calibration: is the model's confidence honest?

ece                  expected calibration error, with equal-width or equal-mass bins
nll, brier           calibration scores that need no bins
fit_temperature      the one number T that divides the logits, fitted by minimising NLL
bootstrap_change     patient-level interval on the change in ECE and NLL after scaling
reliability_diagram  accuracy against confidence per bin, before and after scaling
calibrate_run        the full Step 7 table for one run folder

Runs on CPU in seconds: it only reads logits_{val,test}.pt written by train.fit.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.evaluate import load_split


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15, mode: str = "width") -> float:
    """Average |accuracy - confidence| over confidence bins, weighted by bin size. 0 is perfect.

    mode="width" splits (0, 1] into equal ranges; mode="mass" gives each bin the same number of scans.
    """
    return float(sum(n / len(labels) * abs(acc - conf) for n, conf, acc in _bins(probs, labels, bins, mode)))


def _bins(probs: np.ndarray, labels: np.ndarray, bins: int, mode: str = "width") -> list[tuple]:
    """(number of scans, mean confidence, accuracy) for every non-empty bin."""
    confidence = probs.max(1)
    correct = probs.argmax(1) == labels
    if mode == "width":
        # Each bin includes its upper edge, so a confidence of exactly 1.0 lands in the last bin.
        groups = [np.flatnonzero((confidence > i / bins) & (confidence <= (i + 1) / bins)) for i in range(bins)]
    elif mode == "mass":
        groups = np.array_split(np.argsort(confidence), bins)
    else:
        raise ValueError(f"mode must be 'width' or 'mass', not {mode!r}")
    return [(len(g), confidence[g].mean(), correct[g].mean()) for g in groups if len(g)]


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-probability of the true class."""
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared distance between the probabilities and the one-hot truth (summed over classes)."""
    return float(((probs - np.eye(probs.shape[1])[labels]) ** 2).sum(1).mean())


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, balanced: bool = False) -> float:
    """T > 0 minimising NLL. Only ever call this on validation logits.

    balanced=True weights every class equally, to test whether the class mix explains T.
    """
    weight = 1 / torch.bincount(labels).double() if balanced else None
    logits = logits.double()
    log_t = torch.zeros(1, dtype=torch.double, requires_grad=True)   # log keeps T positive
    optimizer = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(logits / log_t.exp(), labels, weight=weight)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_t.exp().item()


def bootstrap_change(run_dir, split: str, T: float, n: int = 1000, seed: int = 42, bins: int = 15) -> dict:
    """95% interval on (after - before) for ECE and NLL, resampling whole patients.

    Use the differences: ECE itself is biased upwards by resampling.
    """
    d = torch.load(Path(run_dir) / f"logits_{split}.pt")
    y = d["labels"].numpy()
    before = d["logits"].double().softmax(1).numpy()
    after = (d["logits"].double() / T).softmax(1).numpy()
    patients = load_split(run_dir, split)["patients"]
    members = [np.flatnonzero(patients == g) for g in np.unique(patients)]
    rng = np.random.default_rng(seed)
    ece_diff, nll_diff = [], []
    for _ in range(n):
        i = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        ece_diff.append(ece(after[i], y[i], bins) - ece(before[i], y[i], bins))
        nll_diff.append(nll(after[i], y[i]) - nll(before[i], y[i]))
    return {"split": split,
            "ece_change_low": np.percentile(ece_diff, 2.5), "ece_change_high": np.percentile(ece_diff, 97.5),
            "share_ece_worse": np.mean(np.array(ece_diff) > 0),
            "nll_change_low": np.percentile(nll_diff, 2.5), "nll_change_high": np.percentile(nll_diff, 97.5)}


def reliability_diagram(ax, probs: np.ndarray, labels: np.ndarray, title: str, bins: int = 15,
                        low: float = 0.2) -> None:
    """Dots are bins, sized by scans. Below the diagonal means over-confident."""
    rows = _bins(probs, labels, bins)
    n, conf, acc = (np.array(c) for c in zip(*rows))
    ax.plot([low, 1], [low, 1], "--", color="gray", lw=1)
    ax.vlines(conf, conf, acc, color="crimson", lw=1.5, alpha=0.7)
    ax.scatter(conf, acc, s=15 + 250 * np.sqrt(n / n.max()), color="#2f5bd3", alpha=0.8, zorder=3)
    ax.set(xlim=(low, 1.01), ylim=(low, 1.01), xlabel="confidence", ylabel="accuracy",
           title=f"{title}\nECE {ece(probs, labels, bins):.4f}")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)


def calibrate_run(run_dir, figure_path=None, bins: int = 15) -> tuple[float, pd.DataFrame]:
    """Fit T on validation, score both splits before and after. Returns (T, table)."""
    saved = {s: torch.load(Path(run_dir) / f"logits_{s}.pt") for s in ("val", "test")}
    T = fit_temperature(saved["val"]["logits"], saved["val"]["labels"])

    rows, panels = [], []
    for split, d in saved.items():
        y = d["labels"].numpy()
        for name, t in (("before", 1.0), ("after", T)):
            p = (d["logits"].double() / t).softmax(1).numpy()
            assert (p.argmax(1) == d["logits"].argmax(1).numpy()).all(), "scaling must not change predictions"
            rows.append({"split": split, "scaling": name, "T": round(t, 4),
                         "accuracy": (p.argmax(1) == y).mean(), "mean_confidence": p.max(1).mean(),
                         "ece_width": ece(p, y, bins), "ece_mass": ece(p, y, bins, "mass"),
                         "nll": nll(p, y), "brier": brier(p, y)})
            panels.append((p, y, f"{split.capitalize()}, T = {t:.2f}"))

    if figure_path:
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 10))
        for ax, (p, y, title) in zip(axes.flat, panels):
            reliability_diagram(ax, p, y, title, bins)
        fig.tight_layout()
        Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    return T, pd.DataFrame(rows).round(4)
