# oct-calibration

**Can a retinal OCT classifier tell when it's likely to be wrong?**

This project trains a classifier for retinal OCT scans (CNV, DME, drusen, normal) and then tests whether its **confidence can be trusted**: are the scans it's unsure about really the ones it gets wrong? If they are, the model could answer the scans it's sure about and flag the rest for an ophthalmologist.

It's the imaging half of a broader question I study: how to show model uncertainty to the people who rely on it. The other half is confidence cues on ASR captions for Deaf and hard-of-hearing viewers.

> **Status: work in progress (Step 3 of 11 done).** There's no model or results yet. This README will open with the headline result once Step 8 (selective prediction) is done.

## Plan

The project follows the 11 steps in [`docs/OCT-Build-Guide.md`](docs/OCT-Build-Guide.md):

| Step | What | Status |
|---:|---|:---:|
| 1 | Workspace: repo, Kaggle notebook, dataset access | ✅ |
| 2 | Get the data and explore it | ✅ |
| 3 | Split by patient (no leakage) | ✅ |
| 4 | Data pipeline and augmentation | ⏳ |
| 5 | Baseline: ImageNet-pretrained ResNet-18 | |
| 6 | Evaluation with bootstrap confidence intervals | |
| 7 | Calibration: ECE, reliability diagrams, temperature scaling | |
| 8 | Selective prediction: risk–coverage curve | |
| 9 | Grad-CAM on confident mistakes | |
| 10 | One extension (e.g. deep ensemble) | |
| 11 | Write-up | |

## Data

**Kermany et al. 2018, OCT2017, Mendeley Data v3** ([doi:10.17632/rscbjbr9sj.3](https://data.mendeley.com/datasets/rscbjbr9sj/3)), licensed CC BY 4.0. This is the release used in the [*Cell* paper](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5). The popular Kaggle copy (`paultimothymooney/kermany2018`) is the older v2, with fewer NORMAL scans and 242 test scans per class. I use a private Kaggle upload of v3 instead.

| Class | Train scans | Train patients | Test scans | Test patients |
|---|---:|---:|---:|---:|
| CNV | 37,205 | 791 | 250 | 178 |
| DME | 11,348 | 709 | 250 | 167 |
| DRUSEN | 8,616 | 713 | 250 | 169 |
| NORMAL | 51,140 | 3,434 | 250 | 171 |
| **Total** | **108,309** | **4,785** | **1,000** | **635** |

What exploring it showed (details in [`NOTES.md`](NOTES.md), tables in [`results/`](results/)):

- **Imbalanced training set.** NORMAL has about 5.9× as many scans as DRUSEN, and the test set is balanced, so macro-F1 matters more than accuracy.
- **No patient appears in both train and test**, but 787 patients have scans in more than one class. Splits must group by patient ID only.
- **Patients contribute very unevenly:** 1 to 813 train scans each, with a median of 10.
- **Image sizes vary:** 8 sizes, mostly 512×496, 768×496, 1536×496 and 512×512, which matters when resizing.

| CNV | DME |
|---|---|
| ![CNV sample grid](figures/step2/grid_CNV.png) | ![DME sample grid](figures/step2/grid_DME.png) |
| **DRUSEN** | **NORMAL** |
| ![DRUSEN sample grid](figures/step2/grid_DRUSEN.png) | ![NORMAL sample grid](figures/step2/grid_NORMAL.png) |

## Patient split

Saved as [`splits/oct_split_fold0.csv`](splits/oct_split_fold0.csv) (`path, label, patient, split`) by `src.data.patient_split`. About 10% of training **patients** go to validation, with class shares kept equal (`StratifiedGroupKFold`, no shuffling). No patient appears in two splits.

| Split | Scans | Patients |
|---|---:|---:|
| train | 97,479 | 4,303 |
| val | 10,830 | 482 |
| test | 1,000 | 635 |

A fixed random seed was **not** enough to reproduce the split: with shuffling on, Kaggle's older scikit-learn gave a different and badly balanced validation set (42.5% CNV vs 34.4% in train). See [`results/step3_split_comparison.csv`](results/step3_split_comparison.csv) and [`NOTES.md`](NOTES.md).

## Repository layout

```
docs/       build guide and research roadmap
figures/    figures used in NOTES.md and this README
results/    small CSV tables of every result (counts, metrics), one file per step
splits/     the saved patient split that every step from 3 on reads
notebooks/  thin notebooks that clone the repo and call src/
src/        project code (the real code lives here, not in notebooks)
runs/       checkpoints, logits and run outputs (gitignored)
NOTES.md    daily log: what I tried, what broke, what I learned
```

## Running it

The exploration runs in a Kaggle notebook with the v3 dataset attached. Full reproduction commands will be added as the `src/` code lands (Step 3 onward).

## References

- Kermany, D. S. et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell 172(5).
- Guo, C. et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
- Geifman, Y. & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks.* NeurIPS.
