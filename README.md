# Still — stress-related text classification

Local Flask demo using TF–IDF and a binary classifier. English Reddit training
data does not establish clinical validity or reliability for other populations,
languages, or short chat messages. Displayed probabilities are uncalibrated model
estimates; they are not stress severity scores.

## Run

```powershell
.venv/Scripts/python.exe -m pip install -r requirement.txt
.venv/Scripts/python.exe app.py
```

Open http://127.0.0.1:5000. This uses Flask's local development server.

## Reproduce training

Official data: https://www.cs.columbia.edu/~eturcan/data/dreaddit.zip
The downloaded archive is in `data/dreaddit.zip`.

```powershell
.venv/Scripts/python.exe train_model.py
.venv/Scripts/python.exe -m unittest -v
```

The search compares 12 TF–IDF/logistic regression/naive Bayes configurations
using five-fold stratified, post-grouped cross-validation. Duplicate normalized
texts are connected to their post groups to prevent fold leakage. The vocabulary
is fitted inside each fold. Selection uses macro F1 to give both classes equal
weight. Test data never selects hyperparameters or thresholds. Selected settings
are refitted on all 2,838 training rows. Retraining replaces the active artifact;
the original is retained as `model/stress_model.original.joblib`.

## Audit results

Full metrics and confusion matrices: `model/metrics.json`.

| Official test set (715 rows) | Original NB | Selected logistic regression |
| --- | ---: | ---: |
| Accuracy | 71.89% | 72.31% |
| Macro F1 | 0.7113 | 0.7217 |
| Stress precision | 68.18% | 71.54% |
| Stress recall | 85.37% | 76.96% |
| Non-stress recall | 57.51% | 67.34% |

This is a small observed improvement in balanced performance, with reduced stress
recall. It is not evidence of statistical significance. The selected CV macro F1
is 0.7595. Three normalized test texts also occur in training; excluding them
gives 712 test rows and macro F1 of 0.7127 (original) versus 0.7232 (selected).
The original notebook already evaluated the test set, so future experiments
should use a fresh external evaluation set before claiming generalization gains.

The original artifact was saved with scikit-learn 1.6.1 while the installed
environment is 1.9.1. The replacement was trained in the pinned current
environment. Load only trusted joblib files. The app does not persist submitted
text, but server infrastructure may have its own logging policies.

UI verification: Flask rendering and request tests are provided. Automated
browser visual inspection was unavailable in this environment.