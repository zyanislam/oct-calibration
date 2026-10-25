# ---
# Kaggle notebook: zyanislam/oct-calibration
# https://www.kaggle.com/code/zyanislam/oct-calibration
# Synced from the latest saved Kaggle version on Sun 25 Oct 2026.
# Every code and markdown cell, in order. Git history of this file = the notebook's history:
#   git log -p notebooks/kaggle_cells.py
# Opens as a notebook in VS Code / JupyterLab ("# %%" marks a cell).
# ---

# %% [markdown]
# # **Clone the repo and Import the code**

# %%  [code cell 1]
BRANCH = "step4-pipeline"

!rm -rf /tmp/oct-calibration && git clone -q -b {BRANCH} https://github.com/zyanislam/oct-calibration.git /tmp/oct-calibration
import sys; sys.path.insert(0, "/tmp/oct-calibration")
from src.data import build_index, patient_split, check_no_leak

# %% [markdown]
# # **Build The Index**

# %%  [code cell 2]
from pathlib import Path

ROOT = Path("/kaggle/input/datasets/zyanislam/kermany-retinal-oct-v3")

index, problems = build_index(ROOT)
print("Images:", len(index))
print({k: len(v) for k, v in problems.items()})
index.head()

# %% [markdown]
# # **Split by patient**

# %%  [code cell 3]
split, removed = patient_split(index, val_fraction=0.1, fold=0)

print("Train patients removed because they are also in test:", removed)
print(split.groupby("split").agg(scans=("path", "size"), patients=("patient", "nunique")))
print(split.groupby(["split", "class"]).size().unstack())

# %% [markdown]
# # **Save the split**

# %%  [code cell 4]
out = split[["path", "class", "patient", "split"]].rename(columns={"class": "label"})
out.to_csv("/kaggle/working/oct_split_fold0.csv", index=False)
print("Saved", len(out), "rows")

# %% [markdown]
# # **Testing on Img before Step 4**

# %%  [code cell 5]
import torch
import matplotlib.pyplot as plt
from torchvision.io import decode_image
from torchvision.transforms import v2

plt.rcParams["savefig.bbox"] = "tight"
torch.manual_seed(1)

img = decode_image(str(ROOT / "test/CNV/CNV-1032178-1.jpeg"))
print(f"{type(img) = }, {img.dtype = }, {img.shape = }")

plt.figure(figsize=(8, 4))
plt.imshow(img[0], cmap="gray")
plt.title(f"CNV-1032178-1  {img.shape[2]}×{img.shape[1]}")
plt.axis("off")
plt.show()

# %% [markdown]
# # **Exp 1**

# %%  [code cell 6]
import torchvision.transforms.v2.functional as F

def pad_to_square(x):
    """Pad top and bottom (or left and right) with black so height == width."""
    _, h, w = x.shape
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return F.pad(x, [left, top, side - w - left, side - h - top], fill=0)

resize = v2.Resize((224, 224), antialias=True)
squashed = resize(img)
padded = resize(pad_to_square(img))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={"width_ratios": [2.06, 1, 1]})
for a, x, t in zip(ax, [img, squashed, padded],
                   [f"Original {img.shape[2]}×{img.shape[1]}", "A · Resize (squashed)", "B · Pad, then resize"]):
    a.imshow(x[0], cmap="gray"); a.set_title(t); a.axis("off")
plt.show()

# %% [markdown]
# # **Exp 2**

# %%  [code cell 7]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def show(name, x):
    print(f"{name:<12} shape={tuple(x.shape)} dtype={x.dtype} min={x.min().item():.3f} max={x.max().item():.3f}")

show("original", img)
x = v2.Grayscale(num_output_channels=3)(img);   show("3 channels", x)
x = v2.Resize((224, 224), antialias=True)(x);   show("resized", x)
x = v2.ToDtype(torch.float32, scale=True)(x);   show("float", x)
x = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)(x); show("normalised", x)

# %% [markdown]
# # **Exp 3**

# %%  [code cell 8]
train_tf = v2.Compose([
    v2.Grayscale(num_output_channels=3),
    v2.Resize((224, 224), antialias=True),
    v2.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(10),
    v2.ColorJitter(brightness=0.1, contrast=0.1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def unnormalise(t):
    """Undo Normalize so the image displays with its real brightness."""
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * s + m).clamp(0, 1)

torch.manual_seed(1)
fig, ax = plt.subplots(2, 4, figsize=(14, 7))
for a in ax.flat:
    a.imshow(unnormalise(train_tf(img))[0], cmap="gray")
    a.axis("off")
fig.suptitle("8 random training augmentations")
plt.show()

# %% [markdown]
# # **Check**

# %%  [code cell 9]
import time
from collections import Counter
from src.dataset import make_loaders

loaders = make_loaders(ROOT, batch_size=64, num_workers=4)
print("sizes:", {k: len(v.dataset) for k, v in loaders.items()})
assert [len(loaders[s].dataset) for s in ("train", "val", "test")] == [97479, 10830, 1000]

it = iter(loaders["train"])
t = time.time(); x, y, idx = next(it); first = time.time() - t
t = time.time(); next(it); second = time.time() - t
print("batch:", tuple(x.shape), x.dtype, "labels:", sorted(set(y.tolist())),
      f"| first batch {first:.2f}s, next {second:.2f}s")
assert x.shape == (64, 3, 224, 224)

val, train = loaders["val"].dataset, loaders["train"].dataset
assert torch.equal(val[0][0], val[0][0]), "val must not be augmented"
print("val deterministic: True | train augmented:", not torch.equal(train[0][0], train[0][0]))

sample = val.table.groupby("label").head(100)
print("channels in 400 val scans:", dict(Counter(decode_image(str(val.image_root / p)).shape[0] for p in sample.path)))

# %% [markdown]
# # **Loader speed**

# %%  [code cell 10]
it = iter(loaders["train"]); next(it)          # skip the startup batch
t = time.time()
for _ in range(50): next(it)
per_batch = (time.time() - t) / 50
print(f"{per_batch:.3f}s per batch → {64 / per_batch:.0f} images/s")

# %%  [code cell 11]
# Is the slow part reading the JPEG, or transforming it?
from src.dataset import build_transforms
from torchvision.io import ImageReadMode

sample = loaders["train"].dataset.table.sample(200, random_state=0)
paths = [str(ROOT / p) for p in sample.path]
tf = build_transforms(train=True)

t = time.time()
imgs = [decode_image(p, mode=ImageReadMode.GRAY) for p in paths]
decode = (time.time() - t) / len(paths)

t = time.time()
for im in imgs: tf(im)
transform = (time.time() - t) / len(paths)

print(f"decode    {decode * 1000:.1f} ms per scan")
print(f"transform {transform * 1000:.1f} ms per scan")
print(f"total     {(decode + transform) * 1000:.1f} ms → about {4 / (decode + transform):.0f} images/s with 4 workers")
