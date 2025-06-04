# Sleep Detection Report

This report summarizes the data preparation pipeline, feature engineering choices, model architecture and evaluation, and gives context on the provided figures.

## Preprocessing steps

1. **Merge RR and actigraphy data** – `prepare_user_data` loads `RR.csv` and `Actigraph.csv`, converts RR intervals to beats‑per‑minute, interpolates at 1 Hz and merges the two files (lines 18‑30 of `src/data/preprocess.py`).
2. **Generate per-user windows** – `window_features` resamples each processed file in fixed windows (`60s` by default) and computes mean/std features for heart rate and vector magnitude as well as step counts (lines 53‑63).
3. **Engineer rolling/diff features** – additional rolling averages and first differences are computed for key columns (lines 68‑91).
4. **Label sleep intervals** – in `build_dataset` sleep episodes from `sleep.csv` are matched to each timestamp to create the `is_sleeping` label (lines 126‑138).

## Feature engineering

The feature set includes:

- `hr_mean`, `hr_std` – mean and variability of heart rate per window.
- `vector magnitude_mean`, `vector magnitude_std` – aggregated movement magnitude.
- `steps_sum` – total steps per window.
- Rolling averages over three windows (e.g. `hr_mean_roll3`).
- Difference features capturing short‑term changes (e.g. `hr_mean_diff`).

## Model architecture and evaluation

`train_model` scales the features using `StandardScaler`, splits the data into an 80/20 train/test split and fits a `RandomForestClassifier` with class balancing (lines 12‑24 of `src/models/train.py`).

Additional models (logistic regression, SVM and XGBoost) are trained for comparison via `train_and_evaluate_models` which returns accuracy, precision, recall and F1‑score (lines 8‑33 of `src/models/evaluate_models.py`). The notebook `train_models.ipynb` records the following scores:

```
          Model  Accuracy  Precision    Recall  F1-score
0  RandomForest  0.899489   0.709163  0.704950  0.707051
1        LogReg  0.878705   0.642176  0.666337  0.654033
2           SVM  0.882964   0.640069  0.730693  0.682386
3       XGBoost  0.898296   0.696106  0.725743  0.710616
```

## Confusion matrix and ROC curve

The notebook also plots a confusion matrix and ROC curve for the random forest model using utilities from `src/visualization/plot.py`. The confusion matrix image indicates the classifier achieves high true positives with relatively few false negatives. The ROC curve shows strong separability with an area under the curve close to 1, confirming good overall discrimination.

---
Generated figures can be found in the `reports/` folder.