"""Step 9 Grad-CAM: where on the scan did the model look?

GradCAM       hooks on a layer; returns a heatmap for any class of one scan
select_scans  the three panels (confident right, confident wrong, least confident) from saved logits
plot_panel    each scan above its heatmap
gradcam_run   saves the panels and the size-shortcut check for one run folder

A heatmap shows where the model looked, not why it was right. Runs on CPU.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2

from src.calibrate import fit_temperature
from src.data import CLASSES
from src.dataset import DEFAULT_SPLIT, build_transforms
from src.model import build_model


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model.eval()
        layer.register_forward_hook(lambda _, __, output: setattr(self, "activations", output))
        layer.register_full_backward_hook(lambda _, __, output: setattr(self, "gradients", output[0]))

    def __call__(self, image: torch.Tensor, target: int | None = None) -> tuple[np.ndarray, int]:
        """image is one normalised (3, H, W) scan. Returns (heatmap in 0-1 at input size, target)."""
        logits = self.model(image[None])
        target = int(logits.argmax(1)) if target is None else target
        self.model.zero_grad()
        logits[0, target].backward()
        weights = self.gradients.mean((2, 3), keepdim=True)             # one weight per feature map
        cam = nn.functional.relu((weights * self.activations).sum(1, keepdim=True))   # what argues for the class
        cam = nn.functional.interpolate(cam, image.shape[1:], mode="bilinear", align_corners=False)[0, 0].detach()
        return (cam / cam.max().clamp(min=1e-8)).numpy(), target


def select_scans(run_dir, split: str = "test", n: int = 6, seed: int = 42, coverage: float = 0.90,
                 split_csv=DEFAULT_SPLIT) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Scan table with prediction and calibrated confidence, plus row indices for the three panels.

    The threshold comes from validation. Confident-wrong takes the most confident n, and
    confident-right a random n; test has exactly 6 confident mistakes, so that panel is all of them.
    """
    run_dir = Path(run_dir)
    val = torch.load(run_dir / "logits_val.pt")
    T = fit_temperature(val["logits"], val["labels"])
    threshold = np.quantile((val["logits"].double() / T).softmax(1).max(1).values.numpy(), 1 - coverage)
    saved = torch.load(run_dir / f"logits_{split}.pt")
    table = pd.read_csv(split_csv)
    table = table[table["split"] == split].reset_index(drop=True).iloc[saved["rows"].numpy()].reset_index(drop=True)
    probs = (saved["logits"].double() / T).softmax(1).numpy()
    table["y"], table["pred"], table["confidence"] = saved["labels"].numpy(), probs.argmax(1), probs.max(1)
    sure, right = table.confidence >= threshold, table.pred == table.y
    wrong = table[sure & ~right].sort_values("confidence", ascending=False).index.to_numpy()
    rng = np.random.default_rng(seed)
    return table, {"confident_right": rng.choice(table[sure & right].index.to_numpy(), n, replace=False),
                   "confident_wrong": wrong[:n],
                   "least_confident": table.sort_values("confidence").index.to_numpy()[:n]}


def plot_panel(cam: GradCAM, table: pd.DataFrame, rows, image_root, title: str, path=None):
    """One column per scan: the model's input above the same scan with heat for the predicted class."""
    transform, resize = build_transforms(False), v2.Resize((224, 224), antialias=True)
    fig, axes = plt.subplots(2, len(rows), figsize=(2.6 * len(rows), 6), squeeze=False)
    for j, r in enumerate(rows):
        row = table.loc[r]
        raw = decode_image(str(Path(image_root) / row.path), mode=ImageReadMode.GRAY)
        heat, _ = cam(transform(raw))
        shown = resize(raw)[0].numpy()
        axes[0, j].imshow(shown, cmap="gray")
        axes[0, j].set_title(f"true {CLASSES[row.y]}\nsaid {CLASSES[row.pred]}, {row.confidence:.1%}", fontsize=10)
        axes[1, j].imshow(shown, cmap="gray")
        axes[1, j].imshow(heat, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def gradcam_run(run_dir, image_root, out_dir, dme_512_paths=(), n: int = 6, seed: int = 42) -> pd.DataFrame:
    """Saves the three test panels, plus a validation panel for the given 512x512 DME scans.

    Returns the table of scans shown. dme_512_paths needs the original image sizes, which the
    256x256 copy no longer has, so the caller finds them.
    """
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(Path(run_dir) / "best.pt", map_location="cpu"))
    cam = GradCAM(model, model.layer4)
    out_dir = Path(out_dir)
    table, panels = select_scans(run_dir, n=n, seed=seed)
    titles = {"confident_right": "Confident and right", "confident_wrong": "Confident but wrong",
              "least_confident": "Least confident"}
    shown = []
    for name, rows in panels.items():
        plot_panel(cam, table, rows, image_root, f"{titles[name]} (test)", out_dir / f"{name}.png")
        shown.append(table.loc[rows, ["path", "y", "pred", "confidence"]].assign(panel=name))
    if len(dme_512_paths):
        val, _ = select_scans(run_dir, "val", n=n, seed=seed)
        rows = val[val.path.isin(dme_512_paths)].index.to_numpy()
        rows = np.random.default_rng(seed).choice(rows, n, replace=False)
        plot_panel(cam, val, rows, image_root, "DME at 512×512 (validation)", out_dir / "dme_512.png")
        shown.append(val.loc[rows, ["path", "y", "pred", "confidence"]].assign(panel="dme_512"))
    return pd.concat(shown).round(4)
