# Documentation — Human Activity Recognition

This document describes the dataset, algorithms, evaluation protocols, packaging, and remaining opportunities.

---

## 1. Problem statement

Given a fixed-length window of smartphone inertial measurements, predict one of six activities:

1. Walking  
2. Walking upstairs  
3. Walking downstairs  
4. Sitting  
5. Standing  
6. Laying  

Inputs are **not** raw time series. The UCI authors transform each 2.56-second window into a **561-dimensional feature vector** of time- and frequency-domain statistics.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | [UCI HAR Using Smartphones](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) |
| Volunteers | 30 adults (ages 19–48) — 21 train / 9 test |
| Device | Samsung Galaxy S II on the waist |
| Sensors | 3-axis accelerometer + 3-axis gyroscope @ 50 Hz |
| Window | 2.56 s, 50% overlap (128 samples) |
| Features | 561 engineered variables |
| Train | 7,352 windows (`X_train.txt`, `y_train.txt`, `subject_train.txt`) |
| Test | 2,947 windows (`X_test.txt`, `y_test.txt`, `subject_test.txt`) |

Fetch any missing files with:

```bash
python scripts/download_data.py
```

### Feature naming caveat

A few UCI feature names are duplicated. `har.data.load_feature_names` appends `_1`, `_2`, … so pandas columns stay unique.

---

## 3. Algorithms

### 3.1 From-scratch k-Nearest Neighbors

`CustomKNN` in `src/har/knn.py` — Euclidean distance, majority vote, nearest-neighbor tie-break, vectorized with `||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b`. Matches `sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)`.

### 3.2 Classical model zoo

`build_model_zoo()` includes KNN variants, logistic regression, linear/RBF SVM, PCA+SVM, Random Forest, Extra Trees, Hist Gradient Boosting, and calibrated logistic regression.

### 3.3 Sitting / standing specialist & cascade

Most residual errors are **sitting ↔ standing**. `build_sitting_standing_specialist()` trains a calibrated RBF SVM on labels `{4,5}` only.

`CascadedHARClassifier`:

1. Predict with a multiclass base (default: Hist Gradient Boosting).  
2. When the base predicts sitting or standing, override with the specialist.

On the official test set the cascade reaches the **lowest sit↔stand error (5.7%)** among compared models, at a small cost versus plain logistic regression’s top overall accuracy.

### 3.4 Feature importance

Extra Trees consistently ranks gravity / orientation channels highest (`tGravityAcc-*`, `angle(*,gravityMean)`), matching the physics of posture.

### 3.5 Raw-IMU 1D CNN

`src/har/cnn.py` trains a compact PyTorch CNN on the UCI **Inertial Signals** tensors:

- Input shape: `(batch, 9, 128)` — body acc, body gyro, total acc × {x,y,z}
- 3× Conv1d + BatchNorm + pooling → AdaptiveAvgPool → Linear(6)
- Trained with AdamW; best validation checkpoint restored

Download raw windows (not committed — ~182MB):

```bash
python scripts/download_data.py --with-raw
python scripts/run_deep_baseline.py
```

**Takeaway:** the CNN reaches ~94.6% official / ~92.8% LOSO — close to logistic regression on engineered features — while learning directly from waveforms. It still struggles more on sitting ↔ standing, where gravity-orientation features help classical models.

---

## 4. Evaluation protocols

Implemented in `src/har/evaluate.py` and exposed via `scripts/run_benchmark.py --protocol …`.

| Protocol | Command | What it measures |
|---|---|---|
| `holdout` | random 70/30 of train windows | Optimistic (subject leakage) |
| `official` | UCI train → UCI test | Generalization to held-out people |
| `loso` | leave-one-subject-out on 21 train volunteers | Subject-independent CV |

### Reported headline numbers

- Official test best: **Logistic Regression 95.55%**
- LOSO best: **Logistic Regression 93.65%** (mean subject acc 0.936 ± 0.075)
- Random hold-out can still show ~99% — treat it as a teaching baseline only

Metrics: accuracy, weighted/macro F1, and **sitting_standing_error** (fraction of true sit/stand windows confused with the other posture).

---

## 5. Packaging & tooling

| Piece | Role |
|---|---|
| `src/har/` | Installable library (`pip install -e .`) |
| `scripts/export_model.py` | Train + `joblib` export |
| `scripts/predict.py` | CLI inference on a feature matrix |
| `scripts/run_deep_baseline.py` | Features vs 1D CNN comparison |
| `app/streamlit_app.py` | Interactive demo |
| `tests/` | pytest (KNN parity, data, eval, CNN) |
| `.github/workflows/ci.yml` | smoke + pytest + quick official benchmark |
| `pyproject.toml` | package metadata (`[deep]` extra for PyTorch) |

---

## 6. API sketch

```python
from har.data import load_official_splits
from har.evaluate import evaluate_on_official_test, loso_on_train_partition
from har.models import CascadedHARClassifier, build_model_zoo

train, test = load_official_splits()
model = CascadedHARClassifier().fit(train.X, train.y)
pred = model.predict(test.X)

results = evaluate_on_official_test(build_model_zoo(include_custom_knn=False))
```

---

## 7. Limitations

1. **Waist-only sensing** — placement shift changes the feature distribution.  
2. **Handcrafted features** — no raw-signal deep models in this repo yet.  
3. **Subject leakage** if you only use random window splits.  
4. **Sit/stand ambiguity** remains the dominant error mode even with a specialist.

---

## 8. Roadmap status

| Idea | Status |
|---|---|
| Official UCI test evaluation | Done |
| Subject IDs + LOSO | Done |
| Sit/stand specialist + cascade | Done |
| Calibrated probabilities | Done |
| Model export + predict CLI | Done |
| Streamlit demo | Done |
| pytest + GitHub Actions CI | Done |
| README results card + limitations | Done |
| Raw-signal 1D CNN + features comparison | Done |
| Stronger seq models (Transformer / TCN) | Future |
| On-device / streaming demo | Future |

---

## 9. References

1. Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN.  
2. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR.  
3. UCI ML Repository — Human Activity Recognition Using Smartphones Data Set.
