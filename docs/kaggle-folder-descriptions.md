# Kaggle folder descriptions

Copy-paste text for the folder descriptions on the Kaggle dataset `zyanislam/kermany-retinal-oct-v3`. The Kaggle tool can't set descriptions on folders, only on files, so these go in on the website: open the dataset → **Data Card** → **Data Explorer** → click a folder → **Edit** next to its description.

## `train/`

Official v3 training set: 108,309 grayscale OCT scans from 4,785 patients, one folder per class. Files are CLASS-PATIENT-SCAN.jpeg. No training patient appears in test/. Classes are imbalanced (NORMAL 51,140 vs DRUSEN 8,616), so split off validation by patient ID, not by image.

## `test/`

Official v3 test set: 1,000 grayscale OCT scans (250 per class) from 635 patients, none of whom appear in train/. Use only for final evaluation.

## `train/CNV/`

37,205 train scans, choroidal neovascularization (wet AMD): new vessels under the RPE lift it into an irregular hump, often with fluid. File name: CNV-PATIENT-SCAN.jpeg.

## `train/DME/`

11,348 train scans, diabetic macular edema: dark fluid pockets inside a swollen retina, RPE stays flat. File name: DME-PATIENT-SCAN.jpeg.

## `train/DRUSEN/`

8,616 train scans, drusen deposits (early AMD): small regular bumps on the RPE, layers above intact. File name: DRUSEN-PATIENT-SCAN.jpeg.

## `train/NORMAL/`

51,140 train scans, healthy retina: smooth layers and a clean foveal dip. File name: NORMAL-PATIENT-SCAN.jpeg.

## `test/CNV/`

250 test scans, choroidal neovascularization (wet AMD): new vessels under the RPE lift it into an irregular hump, often with fluid. File name: CNV-PATIENT-SCAN.jpeg.

## `test/DME/`

250 test scans, diabetic macular edema: dark fluid pockets inside a swollen retina, RPE stays flat. File name: DME-PATIENT-SCAN.jpeg.

## `test/DRUSEN/`

250 test scans, drusen deposits (early AMD): small regular bumps on the RPE, layers above intact. File name: DRUSEN-PATIENT-SCAN.jpeg.

## `test/NORMAL/`

250 test scans, healthy retina: smooth layers and a clean foveal dip. File name: NORMAL-PATIENT-SCAN.jpeg.
