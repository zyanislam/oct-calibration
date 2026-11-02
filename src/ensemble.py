"""Step 10 deep ensemble: does a team of models beat one temperature number?

load_members      each model's saved logits, checked to cover the same scans in the same order
team_probs        average of the members' probabilities
compare           accuracy, calibration and selective prediction for one set of probabilities
bootstrap_change  patient-level intervals on (second method - first method)
team_size         how the scores change with 1 to 5 members, over every combination
shared_mistakes   the confident mistakes of the single model and the team, with each member's vote
ensemble_run      the full Step 10 tables and plot

Thresholds and T always come from validation. Runs on CPU from logits_{val,test}.pt.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.calibrate import brier, ece, fit_temperature, nll
from src.data import CLASSES
from src.dataset import DEFAULT_SPLIT
from src.selective import aurc, report, risk_coverage


def load_members(run_dirs: dict) -> dict[str, dict]:
    """For val and test: logits as (members, scans, classes), labels, rows and patient IDs.

    run_dirs maps a seed to its run folder. Raises if the members saw different scans.
    """
    table = pd.read_csv(DEFAULT_SPLIT)
    out = {}
    for split in ("val", "test"):
        saved = [torch.load(Path(d) / f"logits_{split}.pt") for d in run_dirs.values()]
        for s in saved:
            assert (s["rows"] == saved[0]["rows"]).all() and (s["labels"] == saved[0]["labels"]).all()
        rows = saved[0]["rows"].numpy()
        part = table[table["split"] == split].reset_index(drop=True).iloc[rows].reset_index(drop=True)
        out[split] = {"logits": torch.stack([s["logits"].double() for s in saved]),
                      "labels": saved[0]["labels"], "patients": part["patient"].to_numpy(),
                      "paths": part["path"].to_numpy()}
    return out


def team_probs(logits: torch.Tensor) -> torch.Tensor:
    """logits is (members, scans, classes). Averages probabilities, not logits (Lakshminarayanan et al. 2017)."""
    return logits.softmax(-1).mean(0)


def compare(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """One row of the comparison table. Scans at or above the threshold are answered."""
    confidence = probs.max(1)
    wrong = probs.argmax(1) != labels
    return {"accuracy": 1 - wrong.mean(), "mistakes": int(wrong.sum()),
            "ece": ece(probs, labels), "nll": nll(probs, labels), "brier": brier(probs, labels),
            "aurc": aurc(confidence, wrong), "threshold": threshold,
            "confident_mistakes": int((wrong & (confidence >= threshold)).sum()),
            **report(confidence, wrong, threshold)}


def bootstrap_change(first: np.ndarray, second: np.ndarray, labels, patients, n: int = 1000, seed: int = 42) -> dict:
    """95% intervals on second - first for accuracy, ECE, NLL and AURC, resampling whole patients."""
    def score(p, y):
        return {"accuracy": (p.argmax(1) == y).mean(), "ece": ece(p, y), "nll": nll(p, y),
                "aurc": aurc(p.max(1), p.argmax(1) != y)}

    members = [np.flatnonzero(patients == g) for g in np.unique(patients)]
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in ("accuracy", "ece", "nll", "aurc")}
    for _ in range(n):
        i = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        a, b = score(first[i], labels[i]), score(second[i], labels[i])
        for k in draws:
            draws[k].append(b[k] - a[k])
    out = {}
    for k, v in draws.items():
        out[f"{k}_change_low"], out[f"{k}_change_high"] = np.percentile(v, [2.5, 97.5])
    return out


def team_size(logits: torch.Tensor, labels: np.ndarray) -> pd.DataFrame:
    """Mean, lowest and highest scores over every team of each size."""
    rows = []
    for size in range(1, len(logits) + 1):
        scores = [{"mistakes": int((p.argmax(1) != labels).sum()), "nll": nll(p, labels),
                   "aurc": aurc(p.max(1), p.argmax(1) != labels)}
                  for p in (team_probs(logits[list(c)]).numpy() for c in combinations(range(len(logits)), size))]
        frame = pd.DataFrame(scores)
        rows.append({"size": size, "teams": len(scores),
                     **{f"{k}_{f}": getattr(frame[k], f)() for k in frame for f in ("mean", "min", "max")}})
    return pd.DataFrame(rows)


def shared_mistakes(single, team, members, labels, paths, patients, single_threshold, team_threshold):
    """Every scan that either method answers wrongly. votes lists each member's prediction, in seed order."""
    answered_s = single.max(1) >= single_threshold
    answered_t = team.max(1) >= team_threshold
    wrong_s, wrong_t = single.argmax(1) != labels, team.argmax(1) != labels
    rows = np.flatnonzero((answered_s & wrong_s) | (answered_t & wrong_t))
    votes = members.argmax(-1).numpy()[:, rows]
    return pd.DataFrame({"path": paths[rows], "patient": patients[rows], "true": [CLASSES[k] for k in labels[rows]],
                         "single_says": [CLASSES[k] for k in single[rows].argmax(1)],
                         "single_confidence": single[rows].max(1),
                         "single_outcome": np.where(answered_s[rows], "answered", "flagged"),
                         "team_says": [CLASSES[k] for k in team[rows].argmax(1)],
                         "team_confidence": team[rows].max(1),
                         "team_outcome": np.where(answered_t[rows], "answered", "flagged"),
                         "members_wrong": (votes != labels[rows]).sum(0),
                         "votes": [" ".join(CLASSES[k] for k in v) for v in votes.T]})


def ensemble_run(run_dirs: dict, baseline, figure_path=None, coverage: float = 0.90) -> dict[str, pd.DataFrame]:
    """run_dirs maps seed to folder; baseline is the seed compared with the team.

    Returns the tables comparison, change, members, team_size and confident_mistakes.
    """
    data = load_members(run_dirs)
    seeds = list(run_dirs)
    base = seeds.index(baseline)
    y = {s: d["labels"].numpy() for s, d in data.items()}

    def scale(logits, T):
        return (logits / T).softmax(-1).numpy()

    T_single = fit_temperature(data["val"]["logits"][base], data["val"]["labels"])
    team_log = {s: team_probs(d["logits"]).log() for s, d in data.items()}
    T_team = fit_temperature(team_log["val"], data["val"]["labels"])
    methods = {"baseline": {s: scale(d["logits"][base], 1) for s, d in data.items()},
               "temperature": {s: scale(d["logits"][base], T_single) for s, d in data.items()},
               "team": {s: scale(team_log[s], 1) for s in data},
               "team + temperature": {s: scale(team_log[s], T_team) for s in data}}
    Ts = {"baseline": 1.0, "temperature": T_single, "team": 1.0, "team + temperature": T_team}
    thresholds = {m: float(np.quantile(p["val"].max(1), 1 - coverage)) for m, p in methods.items()}

    # Each method's own threshold flags a different number of test scans, so also flag the same number.
    flags = int((methods["temperature"]["test"].max(1) < thresholds["temperature"]).sum())
    comparison = []
    for m, p in methods.items():
        for s in ("val", "test"):
            row = {"method": m, "split": s, "T": Ts[m], **compare(p[s], y[s], thresholds[m])}
            if s == "test":
                lowest = np.argsort(p[s].max(1), kind="stable")[:flags]
                row |= {"same_flags": flags, "caught_at_same_flags": int((p[s][lowest].argmax(1) != y[s][lowest]).sum())}
            comparison.append(row)

    change = [{"first": a, "second": b, "split": s,
               **bootstrap_change(methods[a][s], methods[b][s], y[s], data[s]["patients"])}
              for a, b in (("baseline", "team"), ("temperature", "team + temperature")) for s in ("val", "test")]

    members = []
    for k, seed in enumerate(seeds):
        T = fit_temperature(data["val"]["logits"][k], data["val"]["labels"])
        for s, d in data.items():
            before, after = scale(d["logits"][k], 1), scale(d["logits"][k], T)
            members.append({"seed": seed, "split": s, "T": T, "accuracy": (before.argmax(1) == y[s]).mean(),
                            "mistakes": int((before.argmax(1) != y[s]).sum()), "ece": ece(before, y[s]),
                            "ece_after_T": ece(after, y[s]), "nll": nll(before, y[s]),
                            "aurc": aurc(before.max(1), before.argmax(1) != y[s])})

    sizes = pd.concat([team_size(d["logits"], y[s]).assign(split=s) for s, d in data.items()])
    mistakes = shared_mistakes(methods["temperature"]["test"], methods["team + temperature"]["test"],
                               data["test"]["logits"], y["test"], data["test"]["paths"], data["test"]["patients"],
                               thresholds["temperature"], thresholds["team + temperature"])

    if figure_path:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, s, title in zip(axes, ("val", "test"), ("Validation", "Test")):
            for m, color in (("temperature", "#b5650d"), ("team + temperature", "#2f5bd3")):
                conf = methods[m][s].max(1)
                wrong = methods[m][s].argmax(1) != y[s]
                risk = risk_coverage(conf, wrong)
                ax.plot(np.arange(1, len(risk) + 1) / len(risk), risk, color=color, lw=2,
                        label=f"{m}, AURC {risk.mean():.4f}")
            ax.set(xlabel="coverage (share of scans answered)", ylabel="risk (error rate when answered)",
                   title=title, xlim=(0.5, 1), ylim=(0, (y[s] != methods["temperature"][s].argmax(1)).mean() * 1.2))
            ax.grid(alpha=0.25)
            ax.legend()
        fig.tight_layout()
        Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")

    return {"comparison": pd.DataFrame(comparison).round(4), "change": pd.DataFrame(change).round(4),
            "members": pd.DataFrame(members).round(4), "team_size": sizes.round(4),
            "confident_mistakes": mistakes.round(4)}
