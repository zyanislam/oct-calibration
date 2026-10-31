# Results

Small, plain CSV tables of every result, so numbers can be checked or reused without rerunning anything. Files are named by guide step.

| File | What it holds |
|---|---|
| `step2_counts_by_class.csv` | Scans and unique patients per split and class |
| `step2_summary.csv` | Headline Step 2 numbers: patient counts, train–test overlap, scans per patient, class ratio |
| `step2_image_sizes.csv` | Number of scans at each image size (width × height) |
| `step3_split_counts.csv` | Scans and unique patients per split and class in the final split |
| `step3_split_comparison.csv` | Train/val scans, patients and class shares: shuffled split (Kaggle scikit-learn) vs the fixed `shuffle=False` split |
| `step4_size_by_class.csv` | Number of scans of each class at each image size. All 8,659 scans at 512×512 are DME (about 75% of DME), a possible scanner shortcut |
| `step4_image_modes.csv` | Image modes of all 109,309 scans: all grayscale except 2 RGB drusen test scans |
| `step4_loader_speed.csv` | Data loader speed for each setup tried: original vs 256×256 data, Kaggle input disk vs a local copy |
| `step5_training_log.csv` | Per-epoch train loss, val loss, val accuracy, val macro-F1, learning rate and time for the Step 5 baseline |
| `step5_per_class.csv` | Precision, recall and F1 per class for `best.pt` on val and test |
| `step6_metrics.csv` | Test accuracy and macro-F1 with 95% bootstrap intervals (by patient, the headline, and by image), plus validation values |
| `step6_per_class.csv` | Per class and split: precision, recall, F1 and one-vs-rest AUC; test recall and F1 with patient-level 95% intervals |
| `step7_calibration.csv` | Accuracy, mean confidence, ECE (equal-width and equal-mass), NLL and Brier, before and after T, on validation and test |
| `step7_calibration_change.csv` | Patient-level 95% bootstrap intervals on the change in ECE and NLL after scaling |
| `step7_size_by_split.csv` | Scans per split, class and image size. 94% of test scans are 1024×496, a size that is 0.4% of train and validation |

All Step 2 numbers come from `src.data.build_index` on Kermany OCT2017, Mendeley v3 (109,309 images).
