# To do later

Things to come back to that aren't part of the current step.

## Kaggle dataset: make `zyanislam/kermany-retinal-oct-v3` public

- [ ] Add to the description: "Unmodified copy of the OCT part of Mendeley Data v3, re-uploaded for convenience. I am not the original author."
- [ ] Remove the stray `.DS_Store` from the dataset root (publish a new version)
- [ ] Fill in the usability fields: cover image, tags (`medicine`, `image`, `classification`), file descriptions
- [ ] Decide for sure first (Kaggle may not let it go back to private), then Settings → Sharing → Public
- [ ] Link it from the README's Data section

## Notebook tidy-up

- [x] Clone the repo to `/tmp/oct-calibration` in the first cell so Kaggle's output folder holds only results
- [x] Delete cell 7 (duplicates the folder listing in cell 2)
- [ ] Make the Kaggle notebook public when the project is ready to share (Step 11)

## Optional experiments

- [ ] Train the Step 5 baseline on the shuffled (unbalanced) split too, and compare val macro-F1 and the Step 7 temperature T with the fixed split. Shows whether the imbalance changes conclusions. The old split is kept at `runs/step3/oct_split_shuffled_sklearn-kaggle.csv` (not committed).

## Housekeeping

- [ ] Optional: remove the token line from `~/.zsh_history`
