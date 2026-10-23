# Notes

**Code layout rule:** real code lives in `.py` files under `src/`. The notebook only clones the repo and calls those files.

## Daily log

One line a day: what I tried, what broke, what I learned.

- **2026-10-22:** Created the `oct-calibration` repo, Colab notebook (T4) and Kaggle token (Step 1). Downloaded Kermany OCT2017 (Step 2).
- **2026-10-23:** Added the build guide + roadmap to `docs/`, a `.gitignore` (runs/, CellData/, kaggle.json) and this file. Counted images per class (train is very imbalanced: NORMAL 51k vs DRUSEN 8.6k; test is 250 each). Labelling practice: Round 1 3/8 → Round 2 6/8 → Round 3 8/8. Learned to check the RPE line first; CNV was my weak class, now drusen vs CNV is. Round 3 was drusen/CNV heavy and I got all 8: "smooth & sharp vs fuzzy & broken" works.
- **2026-10-24:** Moved Step 2 to a Kaggle notebook (`zyanislam/oct-calibration`). Built an index table (path, split, class, patient, width, height): 0 malformed names, 0 class mismatches, 0 unreadable files. Found the Kaggle `kermany2018` copy is the older v2 (84,484 images, stored twice), so I uploaded my local Mendeley v3 copy as the private dataset `zyanislam/kermany-retinal-oct-v3`. Reran the index on v3: 109,309 images (108,309 train, 1,000 test), all checks 0. Widths vary (512, 768, 1024, 1536; height 496). Patients: 4,785 train, 635 test, 0 in both; 787 span more than one class. Saved a 20-scan grid per class. Next: patients per split, train–test overlap, sample grids.

## Data (Step 2)

Kermany OCT2017, local copy in `CellData/OCT/` (gitignored).

**Dataset version: Mendeley v3** ([doi:10.17632/rscbjbr9sj.3](https://data.mendeley.com/datasets/rscbjbr9sj/3)), the release used in Kermany et al. 2018 (*Cell*). On Kaggle I use my private copy `zyanislam/kermany-retinal-oct-v3` ("Kermany Retinal OCT (Mendeley v3)"), not `paultimothymooney/kermany2018`. That one is the older v2 (Mar 2018): 83,484 train with only 26,315 NORMAL, and 968 test (242 per class). It also contains the whole dataset twice (`OCT2017 /` and `oct2017/OCT2017 /`, with a trailing space in the folder name).

| Class  | Train  | Test |
|--------|-------:|-----:|
| CNV    | 37,205 | 250  |
| DME    | 11,348 | 250  |
| DRUSEN |  8,616 | 250  |
| NORMAL | 51,140 | 250  |

Largest / smallest train class ratio: 

51,140 : 8,616 = 5.935 : 1

- For 1 sample of DRUSEN, there are almost 6 samples from the NORMAL class.

### Patients (v3, from the Kaggle index)

| Split | Unique patients | CNV | DME | DRUSEN | NORMAL |
|-------|----------------:|----:|----:|-------:|-------:|
| train | 4,785 | 791 | 709 | 713 | 3,434 |
| test  |   635 | 178 | 167 | 169 |   171 |

- **Patients in both train and test: 0.** The official v3 test set is already patient-disjoint, so Step 3 removes no one from training (still keep the assert).
- **787 patients have scans in more than one class**, so the Step 3 split must group on patient only, never patient + class.
- Train scans per patient: min 1, median 10, max 813. A few patients contribute hundreds of scans, so the same patient shows up several times in a random sample (e.g. 904464 and 6668596 twice in the CNV grid).
- Image sizes: widths 512, 768, 1024 and 1536, all 496 px tall.

### Sample grids (20 random train scans per class, seed 42)

![CNV grid](figures/step2/grid_CNV.png)

![DME grid](figures/step2/grid_DME.png)

![DRUSEN grid](figures/step2/grid_DRUSEN.png)

![NORMAL grid](figures/step2/grid_NORMAL.png)

What each class looks like:

- CNV: RPE lifted into one big irregular hump with a dark shadow below.
- DME: It has a few dark fluid pockets inside the swollen retina.
- DRUSEN: It contains a row of small, regular bumps on the RPE. The retina above is intact and the fovea is still visible.
- NORMAL: no dark pockets or bumps present insight. Smooth RPE with a clean dip in the center.

## Labelling practice

Can I tell the classes apart by eye? 2 random scans per class, shuffled, from `runs/quiz/quiz.png`.
Write my guess before checking `runs/quiz/answers.json`.

Anatomy reference (normal, DME, CNV, drusen with layers labelled):

![Labelled OCT anatomy](figures/practice/anatomy-labelled.png)

### Round 1 (2026-10-23)

![Round 1 scans](figures/practice/round1.png)

| Scan | My guess | Why (what I saw) | Actual | Right? |
|-----:|----------|------------------|--------|:------:|
| 1 | DME | | DME | ✓ |
| 2 | (skipped) | | DRUSEN | – |
| 3 | DRUSEN | | CNV | ✗ |
| 4 | DME | | CNV | ✗ |
| 5 | DRUSEN | | DRUSEN | ✓ |
| 6 | NORMAL | | NORMAL | ✓ |
| 7 | (skipped) | | NORMAL | – |
| 8 | (skipped) | | DME | – |

Score: 3/8 (3/5 attempted)

Lessons:
- CNV vs DME: DME = many dark fluid pockets *inside* the retina (#1). CNV = the bright bottom layer (RPE) is lifted or broken by a mass *under* the retina, fluid on top is secondary (#4: one cyst, but the RPE below is humped up).
- CNV vs DRUSEN: drusen are small, regular bumps of the RPE with the retina above intact (#5). An irregular, larger RPE elevation with tissue/fluid around it is CNV (#3).
- NORMAL: smooth layers + clean foveal dip (#6, #7). The tilt/curve of the image doesn't matter.
- #2 is labelled DRUSEN but looks almost normal to me. Kermany labels are image-level and can be noisy; remember this for Step 7–8 (confident "mistakes" may be label errors).
- #8 (DME) is low quality and cropped: real data has bad scans too.

### Round 2 (2026-10-23)

![Round 2 scans](figures/practice/round2.png)

From `runs/quiz/round2/quiz.png`. Tip: check the RPE line first.

| Scan | My guess | Why (what I saw) | Actual | Right? |
|-----:|----------|------------------|--------|:------:|
| 1 | DME | | DME | ✓ |
| 2 | DRUSEN | | CNV | ✗ |
| 3 | NORMAL | | NORMAL | ✓ |
| 4 | DRUSEN | | DRUSEN | ✓ |
| 5 | CNV | | DRUSEN | ✗ |
| 6 | NORMAL | | NORMAL | ✓ |
| 7 | CNV | | CNV | ✓ |
| 8 | DME | | DME | ✓ |

Score: 6/8 (up from 3/8). Only error: swapped #2 and #5 (drusen vs CNV).

Lessons:
- #5 is DRUSEN even though the bumps are big and clustered: they are smooth, dome-shaped RPE elevations, the layers above follow them cleanly, and there's no fuzzy tissue or fluid. Size alone doesn't make it CNV.
- #2 is CNV: the RPE band is broken and messy (bright, fuzzy, ill-defined material under the retina) with the retina distorted above. Irregular and blurred, not smooth domes.
- #7: a single large, smooth dome is a pigment epithelial detachment (PED); in this dataset it was labelled CNV.
- #1 is DME but the scan is off-centre (the big dip on the left is the optic disc, not the fovea).
- Drusen vs CNV is my weak spot now. Next round: look for *smooth & sharp* (drusen) vs *fuzzy, broken, with fluid* (CNV).

### Round 3 (2026-10-23)

![Round 3 scans](figures/practice/round3.png)

From `runs/quiz/round3/quiz.png`. Weighted toward drusen vs CNV (not 2 per class). Smooth & sharp domes = drusen; fuzzy, broken RPE, fluid = CNV.

| Scan | My guess | Why (what I saw) | Actual | Right? |
|-----:|----------|------------------|--------|:------:|
| 1 | DRUSEN | | DRUSEN | ✓ |
| 2 | DRUSEN | | DRUSEN | ✓ |
| 3 | DME | | DME | ✓ |
| 4 | CNV | | CNV | ✓ |
| 5 | CNV | | CNV | ✓ |
| 6 | NORMAL | | NORMAL | ✓ |
| 7 | CNV | | CNV | ✓ |
| 8 | DRUSEN | | DRUSEN | ✓ |

Score: 8/8 (3/8 → 6/8 → 8/8). Drusen vs CNV sorted on the drusen/CNV-heavy round.
