"""Index the Kermany OCT2017 (Mendeley v3) images and split them by patient.

build_index    one row per image: path, split, class, patient, width, height  (Step 2)
patient_split  official train -> train/val by patient, test kept as is        (Step 3)
check_no_leak  assert that no patient appears in more than one split
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ("CNV", "DME", "DRUSEN", "NORMAL")
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def _read_size(path: str):
    try:
        # reads the header only, not the pixels
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def build_index(root, read_sizes: bool = True, workers: int = 16, progress: bool = True):
    """Scan root/{train,test}/{CLASS}/CLASS-PATIENT-SCAN.jpeg.

    Returns (df, problems). Paths in df are relative to root, so the same
    table works on Kaggle and on a local copy.
    """
    root = Path(root)
    assert root.is_dir(), f"Directory not found: {root}"

    rows = []
    problems = {"malformed": [], "mismatch": [], "not_a_class": [], "unreadable": []}

    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.startswith("._"):  # macOS metadata files
                continue
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
                continue

            path = Path(dirpath) / name
            rel = str(path.relative_to(root))

            # CNV-1016042-1 -> CNV, 1016042, 1
            parts = path.stem.split("-")
            if len(parts) != 3:
                problems["malformed"].append(rel)
                continue

            name_class, patient, _scan = parts
            folder_class = path.parent.name
            if folder_class not in CLASSES:
                problems["not_a_class"].append(rel)
                continue
            if name_class != folder_class:
                problems["mismatch"].append(rel)

            rows.append({
                "path": rel,
                "split": path.parent.parent.name,  # train / test
                "class": folder_class,
                "patient": patient,  # a label, so kept as a string
            })

    # Sort so the row order (and so every seeded sample) is the same on any machine.
    df = pd.DataFrame(rows).sort_values("path", ignore_index=True)

    if read_sizes:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            sizes = pool.map(_read_size, (str(root / p) for p in df["path"]), chunksize=256)
            if progress:
                from tqdm.auto import tqdm
                sizes = tqdm(sizes, total=len(df), desc="Reading sizes")
            sizes = list(sizes)
        ok = [s is not None for s in sizes]
        problems["unreadable"] = df.loc[[not o for o in ok], "path"].tolist()
        df = df[ok].reset_index(drop=True)
        df["width"] = [s[0] for s in sizes if s is not None]
        df["height"] = [s[1] for s in sizes if s is not None]

    return df, problems


def check_no_leak(df: pd.DataFrame) -> None:
    """Fail loudly if any patient has scans in more than one split."""
    splits_per_patient = df.groupby("patient")["split"].nunique()
    leaked = splits_per_patient[splits_per_patient > 1]
    assert leaked.empty, (
        f"{len(leaked)} patients appear in more than one split, e.g. {list(leaked.index[:5])}"
    )


def patient_split(df: pd.DataFrame, val_fraction: float = 0.1, seed: int = 42):
    """Split the official training images into train and val by patient.

    Every scan of a patient lands on the same side. Classes are kept in similar
    proportions (StratifiedGroupKFold). Train patients who also appear in the
    official test set are removed first. Returns (df, removed_patients).
    """
    assert set(df["split"]) <= {"train", "test"}, f"unexpected splits: {set(df['split'])}"
    df = df.copy()

    # 1. Remove train patients who also appear in the official test set.
    test_patients = set(df.loc[df["split"] == "test", "patient"])
    in_both = (df["split"] == "train") & df["patient"].isin(test_patients)
    removed_patients = df.loc[in_both, "patient"].nunique()
    df = df[~in_both]

    # 2. Hold out about val_fraction of the training patients as validation.
    train = df[df["split"] == "train"]
    n_splits = round(1 / val_fraction)  # 0.1 -> 10 folds, use one as val
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_rows = next(sgkf.split(train, train["class"], groups=train["patient"]))
    df.loc[train.index[val_rows], "split"] = "val"

    check_no_leak(df)
    return df.reset_index(drop=True), removed_patients