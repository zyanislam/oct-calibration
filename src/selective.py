"""Step 8 selective prediction: does low confidence point at the mistakes?

scores          confidence, margin and negative entropy for each scan (higher = more certain)
risk_coverage   error rate among the top-scoring scans, at every coverage
aurc            area under that curve
report          answered and flagged results at one threshold
bootstrap_report  patient-level intervals on that report, threshold held fixed
selective_run   the full Step 8 tables and plot for one run folder

Thresholds always come from validation. Runs on CPU from logits_{val,test}.pt.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.calibrate import fit_temperature
from src.data import CLASSES
from src.evaluate import load_split


def scores(probs: np.ndarray) -> dict[str, np.ndarray]:
    top = np.sort(probs, axis=1)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(1)
    return {"confidence": top[:, -1], "margin": top[:, -1] - top[:, -2], "entropy": -entropy}


def risk_coverage(score: np.ndarray, wrong: np.ndarray) -> np.ndarray:
    """risk[k] is the error rate among the k + 1 highest-scoring scans."""
    ordered = wrong[np.argsort(-score, kind="stable")]
    return np.cumsum(ordered) / np.arange(1, len(wrong) + 1)


def aurc(score: np.ndarray, wrong: np.ndarray) -> float:
    return float(risk_coverage(score, wrong).mean())


def report(confidence: np.ndarray, wrong: np.ndarray, threshold: float) -> dict:
    """Scans at or above the threshold are answered; the rest are flagged."""
    answered = confidence >= threshold
    errors = wrong.sum()
    return {"coverage": answered.mean(),
            "accuracy_all": 1 - wrong.mean(),
            "accuracy_answered": 1 - wrong[answered].mean() if answered.any() else np.nan,
            "accuracy_flagged": 1 - wrong[~answered].mean() if (~answered).any() else np.nan,
            "flagged": int((~answered).sum()),
            "errors": int(errors),
            "errors_caught": int(wrong[~answered].sum()),
            "share_caught": wrong[~answered].sum() / errors if errors else np.nan}


def bootstrap_report(confidence, wrong, patients, threshold: float, n: int = 1000, seed: int = 42) -> dict:
    """95% intervals, resampling whole patients. Covers test noise, not threshold uncertainty."""
    members = [np.flatnonzero(patients == g) for g in np.unique(patients)]
    rng = np.random.default_rng(seed)
    keys = ("coverage", "accuracy_answered", "share_caught")
    draws = {k: [] for k in keys}
    for _ in range(n):
        i = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        r = report(confidence[i], wrong[i], threshold)
        for k in keys:
            draws[k].append(r[k])
    out = {}
    for k in keys:
        out[f"{k}_low"], out[f"{k}_high"] = np.nanpercentile(draws[k], [2.5, 97.5])
    return out


def selective_run(run_dir, figure_path=None, coverages=(0.95, 0.90, 0.80)) -> dict[str, pd.DataFrame]:
    """Returns the tables operating_points, per_class and score_comparison."""
    run_dir = Path(run_dir)
    saved = {s: torch.load(run_dir / f"logits_{s}.pt") for s in ("val", "test")}
    T = fit_temperature(saved["val"]["logits"], saved["val"]["labels"])
    data = {}
    for s, d in saved.items():
        p = (d["logits"].double() / T).softmax(1).numpy()
        y = d["labels"].numpy()
        data[s] = {"y": y, "wrong": p.argmax(1) != y, "scores": scores(p),
                   "patients": load_split(run_dir, s)["patients"]}

    points, per_class = [], []
    for target in coverages:
        threshold = float(np.quantile(data["val"]["scores"]["confidence"], 1 - target))
        for s, d in data.items():
            conf = d["scores"]["confidence"]
            row = {"target_coverage": target, "threshold": threshold, "split": s,
                   **report(conf, d["wrong"], threshold)}
            if s == "test":
                row |= bootstrap_report(conf, d["wrong"], d["patients"], threshold)
            points.append(row)
            for k, c in enumerate(CLASSES):
                per_class.append({"target_coverage": target, "split": s, "class": c,
                                  "flag_rate": (conf[d["y"] == k] < threshold).mean()})

    comparison = pd.DataFrame([{"score": name, "aurc_val": aurc(data["val"]["scores"][name], data["val"]["wrong"]),
                                "aurc_test": aurc(data["test"]["scores"][name], data["test"]["wrong"])}
                               for name in ("confidence", "margin", "entropy")])

    if figure_path:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, (s, d) in zip(axes, data.items()):
            risk = risk_coverage(d["scores"]["confidence"], d["wrong"])
            coverage = np.arange(1, len(risk) + 1) / len(risk)
            errors = d["wrong"].sum()
            best = np.maximum(0, np.arange(1, len(risk) + 1) - (len(risk) - errors)) / np.arange(1, len(risk) + 1)
            ax.axhline(d["wrong"].mean(), ls="--", color="gray", label="random flagging")
            ax.plot(coverage, best, color="#147a4b", label="perfect ranking")
            ax.plot(coverage, risk, color="#2f5bd3", lw=2, label="confidence")
            ax.set(xlabel="coverage (share of scans answered)", ylabel="risk (error rate when answered)",
                   title=f"{'Validation' if s == 'val' else 'Test'} · AURC {risk.mean():.4f}",
                   ylim=(0, d["wrong"].mean() * 1.5))
            ax.grid(alpha=0.25)
            ax.legend()
        fig.tight_layout()
        Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")

    return {"operating_points": pd.DataFrame(points).round(4), "per_class": pd.DataFrame(per_class).round(4),
            "score_comparison": comparison.round(4)}
