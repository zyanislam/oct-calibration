"""Step 6 evaluation: metrics, bootstrap intervals and the confusion matrix.

load_split      probabilities, labels and patient IDs for one split's saved logits
metrics         accuracy, macro-F1, per-class precision / recall / F1 and one-vs-rest AUC
bootstrap       95% intervals by resampling images or whole patients
plot_confusion  counts and % of row, side by side
evaluate_run    the full Step 6 table for one run folder

Runs on CPU in seconds: it only reads logits_{val,test}.pt written by train.fit.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from src.data import CLASSES
from src.dataset import DEFAULT_SPLIT

# Metrics that get a bootstrap interval (decision 2). Precision and AUC are point values only.
CI_METRICS = ["accuracy", "macro_f1"] + [f"{m}_{c}" for m in ("recall", "f1") for c in CLASSES]


def load_split(run_dir, split: str, split_csv=DEFAULT_SPLIT) -> dict:
    """Softmax probabilities, labels and patient IDs. rows index the split's own rows in the CSV."""
    saved = torch.load(Path(run_dir) / f"logits_{split}.pt")
    table = pd.read_csv(split_csv)
    table = table[table["split"] == split].reset_index(drop=True)
    return {
        "probs": saved["logits"].softmax(1).numpy(),
        "labels": saved["labels"].numpy(),
        "patients": table["patient"].to_numpy()[saved["rows"].numpy()],
    }


def metrics(labels: np.ndarray, probs: np.ndarray, auc: bool = True) -> dict:
    """Flat dict: accuracy, macro_f1, precision_CNV, recall_CNV, f1_CNV, auc_CNV, ..."""
    pred = probs.argmax(1)
    k = list(range(len(CLASSES)))
    p, r, f, _ = precision_recall_fscore_support(labels, pred, labels=k, zero_division=0)
    out = {"accuracy": (pred == labels).mean(), "macro_f1": f.mean()}
    for i, c in enumerate(CLASSES):
        out |= {f"precision_{c}": p[i], f"recall_{c}": r[i], f"f1_{c}": f[i]}
        if auc:
            out[f"auc_{c}"] = roc_auc_score(labels == i, probs[:, i])
    return out


def bootstrap(labels: np.ndarray, probs: np.ndarray, groups: np.ndarray | None = None,
              n: int = 1000, seed: int = 42, names=CI_METRICS) -> pd.DataFrame:
    """2.5th and 97.5th percentiles over n resamples.

    groups=None resamples scans; groups=patient IDs draws whole patients with replacement
    and keeps all their scans, because scans of one eye are not independent.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        members = [np.array([i]) for i in range(len(labels))]
    else:
        members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    draws = []
    for _ in range(n):
        idx = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        m = metrics(labels[idx], probs[idx], auc=False)
        draws.append([m[k] for k in names])
    draws = np.array(draws)
    return pd.DataFrame({"metric": names, "ci_low": np.percentile(draws, 2.5, axis=0),
                         "ci_high": np.percentile(draws, 97.5, axis=0)})


def plot_confusion(labels: np.ndarray, probs: np.ndarray, title: str, path=None):
    """Counts on the left, % of the true class (row = recall) on the right."""
    cm = confusion_matrix(labels, probs.argmax(1), labels=range(len(CLASSES)))
    pct = cm / cm.sum(1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, values, fmt, name in ((axes[0], cm, "{:,.0f}", "Counts"), (axes[1], pct, "{:.1f}%", "% of true class")):
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center",
                        color="white" if pct[i, j] > 60 else "black", fontsize=11)
        ax.set_xticks(range(len(CLASSES)), CLASSES)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(name)
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def evaluate_run(run_dir, n: int = 1000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary, per_class).

    summary: accuracy and macro-F1 on test with patient-level (headline) and image-level
    intervals, plus validation point values.
    per_class: precision, recall, F1 and AUC per class and split; test recall and F1
    carry patient-level intervals.
    """
    test, val = load_split(run_dir, "test"), load_split(run_dir, "val")
    point_t = metrics(test["labels"], test["probs"])
    point_v = metrics(val["labels"], val["probs"])
    by_patient = bootstrap(test["labels"], test["probs"], test["patients"], n, seed).set_index("metric")
    by_image = bootstrap(test["labels"], test["probs"], None, n, seed,
                         names=["accuracy", "macro_f1"]).set_index("metric")

    summary = pd.DataFrame([{
        "metric": m, "test": point_t[m],
        "test_ci_low": by_patient.ci_low[m], "test_ci_high": by_patient.ci_high[m],
        "test_ci_low_image": by_image.ci_low[m], "test_ci_high_image": by_image.ci_high[m],
        "val": point_v[m],
    } for m in ("accuracy", "macro_f1")])

    rows = []
    for split, point in (("test", point_t), ("val", point_v)):
        for c in CLASSES:
            row = {"split": split, "class": c}
            for m in ("precision", "recall", "f1", "auc"):
                key = f"{m}_{c}"
                row[m] = point[key]
                if split == "test" and key in by_patient.index:
                    row[f"{m}_ci_low"], row[f"{m}_ci_high"] = by_patient.ci_low[key], by_patient.ci_high[key]
            rows.append(row)
    cols = ["split", "class", "precision", "recall", "recall_ci_low", "recall_ci_high",
            "f1", "f1_ci_low", "f1_ci_high", "auc"]
    return summary.round(4), pd.DataFrame(rows)[cols].round(4)
