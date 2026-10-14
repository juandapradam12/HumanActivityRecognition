# Documentation — Human Activity Recognition

Complete methodology for this repository: problem setup, data, algorithms, evaluation, packaging, and how to reproduce every reported number.

For a lighter tour see [`WALKTHROUGH.md`](WALKTHROUGH.md).  
For intended use / risk notes see [`MODEL_CARD.md`](MODEL_CARD.md).

---

## Table of contents

1. [Problem statement](#1-problem-statement)  
2. [Dataset](#2-dataset)  
3. [Algorithms](#3-algorithms)  
4. [Evaluation protocols](#4-evaluation-protocols)  
5. [Reported results](#5-reported-results)  
6. [Packaging & CLI](#6-packaging--cli)  
7. [Python API](#7-python-api)  
8. [Reproducibility](#8-reproducibility)  
9. [Limitations & future work](#9-limitations--future-work)  
10. [References](#10-references)

---

## 1. Problem statement

**Input:** a fixed-length window of smartphone inertial measurements.  
**Output:** one of six activity labels:

| ID | Activity |
|---:|---|
| 1 | WALKING |
| 2 | WALKING_UPSTAIRS |
| 3 | WALKING_DOWNSTAIRS |
| 4 | SITTING |
| 5 | STANDING |
| 6 | LAYING |

This repo studies the problem in **two input representations**:

| Representation | Shape | Source |
|---|---|---|
| Handcrafted features | `(n, 561)` | UCI time/frequency stats per window |
| Raw IMU tensor | `(n, 9, 128)` | body acc, body gyro, total acc × {x,y,z} |

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | [UCI HAR Using Smartphones](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) |
| Volunteers | 30 adults (19–48 years): **21 train / 9 test** |
| Device | Samsung Galaxy S II, waist-mounted |
| Sensors | 3-axis accelerometer + 3-axis gyroscope @ 50 Hz |
| Window | 2.56 s, 50% overlap → 128 samples |
| Train windows | 7,352 |
| Test windows | 2,947 |

### Files in this repo

| File | Role |
|---|---|
| `X_train.txt` / `X_test.txt` | 561-D feature matrices |
| `y_train.txt` / `y_test.txt` | integer labels `{1..6}` |
| `subject_train.txt` / `subject_test.txt` | volunteer IDs (for LOSO) |
| `features.txt` | feature names |
| `activity_labels.txt` | id → name |
| `data/inertial_signals/*.txt` | raw channels (via `--with-raw`, not committed) |

```bash
python scripts/download_data.py            # tabular + subjects
python scripts/download_data.py --with-raw # + ~182MB IMU tensors
```

### Feature naming caveat

UCI duplicates a few feature names. `har.data.load_feature_names` appends `_1`, `_2`, … so pandas columns stay unique without dropping information.

### Class balance (train)

| Label | Activity | Count |
|---:|---|---:|
| 1 | WALKING | 1,226 |
| 2 | WALKING_UPSTAIRS | 1,073 |
| 3 | WALKING_DOWNSTAIRS | 986 |
| 4 | SITTING | 1,286 |
| 5 | STANDING | 1,374 |
| 6 | LAYING | 1,407 |

---

## 3. Algorithms

### 3.1 From-scratch k-NN (`src/har/knn.py`)

Educational Euclidean k-NN with:

1. Distance to all training points  
2. `k` nearest neighbors (`np.argpartition`)  
3. Majority vote; ties → nearest neighbor’s label  

Vectorized with \(\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2a^\top b\).  
**Parity check:** agrees with `sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)` on tested slices.

### 3.2 Classical model zoo (`src/har/models.py`)

`build_model_zoo()` includes:

- Custom / sklearn KNN (uniform & distance-weighted)  
- Logistic regression (scaled)  
- Linear SVM & RBF SVM  
- PCA(100) + RBF SVM  
- Random Forest / Extra Trees  
- Hist Gradient Boosting  
- Calibrated logistic regression (`CalibratedClassifierCV`)

### 3.3 Sitting / standing specialist & cascade

Hard pair: **sitting ↔ standing**.

- `build_sitting_standing_specialist()` — calibrated RBF SVM on labels `{4,5}`  
- `CascadedHARClassifier` — multiclass base (default HGB); when the base predicts 4 or 5, re-decide with the specialist  

On official test the cascade achieves the **lowest sit↔stand error (~5.7%)** among compared models.

### 3.4 Feature importance

Extra Trees ranks gravity / orientation channels highest:

- `tGravityAcc-energy/mean/min/max`  
- `angle(X|Y|Z, gravityMean)`  

Physically expected: posture is mostly device orientation relative to gravity; locomotion rides on body-motion energy and frequency content.

### 3.5 Raw-IMU 1D CNN (`src/har/cnn.py`)

Compact CPU-friendly network:

```text
Input (B, 9, 128)
  → Conv1d 9→64  + BN + ReLU + MaxPool
  → Conv1d 64→128 + BN + ReLU + MaxPool
  → Conv1d 128→128 + BN + ReLU
  → AdaptiveAvgPool1d(1)
  → Dropout → Linear(6)
```

- Optimizer: AdamW (`lr=1e-3`, `weight_decay=1e-4`)  
- Loss: cross-entropy (labels mapped `{1..6}→{0..5}`)  
- Checkpoint: best validation accuracy within the training partition  

---

## 4. Evaluation protocols

Implemented in `src/har/evaluate.py`, CLI via `scripts/run_benchmark.py`.

| Protocol | Flag | Definition | Use |
|---|---|---|---|
| Hold-out | `--protocol holdout` | Stratified 70/30 of **train windows** | Teaching / sanity only |
| Official | `--protocol official` | UCI train → UCI test (held-out people) | **Primary** |
| LOSO | `--protocol loso` | Leave-one-subject-out on 21 train volunteers | Subject-independence |

### Metrics

- Accuracy  
- Weighted F1 / macro F1  
- **`sitting_standing_error`** — among true sit/stand windows, fraction confused with the other posture  

Deep comparison: `scripts/run_deep_baseline.py` (features zoo + CNN, official + LOSO).

---

## 5. Reported results

### Official UCI test

| Model | Accuracy | Weighted F1 | Sit↔Stand error |
|---|---:|---:|---:|
| Logistic Regression | **0.9555** | 0.9555 | 0.0704 |
| RBF SVM | 0.9549 | 0.9547 | 0.0635 |
| Cascaded HGB + Sit/Stand | 0.9518 | 0.9517 | **0.0567** |
| 1D CNN (raw IMU) | 0.9464 | 0.9458 | 0.1232 |
| Extra Trees | 0.9413 | 0.9409 | 0.0616 |
| Hist Gradient Boosting | 0.9359 | 0.9356 | 0.1026 |
| sklearn KNN (k=5) | 0.8802 | 0.8790 | 0.1457 |

Sitting/standing specialist alone: **94.23%** on sit/stand test windows.

### LOSO (21 training subjects)

| Model | Overall acc. | Mean ± std (subject) | Sit↔Stand error |
|---|---:|---:|---:|
| Logistic Regression | **0.9365** | 0.9355 ± 0.0748 | 0.0831 |
| Hist Gradient Boosting | 0.9293 | 0.9302 ± 0.0749 | 0.1038 |
| 1D CNN (raw IMU) | 0.9278 | 0.9269 ± 0.0887 | 0.1485 |
| Extra Trees | 0.9253 | 0.9254 ± 0.0652 | 0.0981 |

### Interpretation

1. **Engineered features still win slightly** on this dataset — UCI’s feature design is excellent.  
2. **CNN closes most of the gap** with no handcrafted pipeline.  
3. **Locomotion is easy; posture is hard** — especially for the CNN, which lacks explicit gravity-angle features.  
4. Quote **official / LOSO**, not random hold-out, in any write-up.

Figures: `docs/assets/official_leaderboard.png`, `official_confusion.png`, `loso_leaderboard.png`, `features_vs_cnn.png`.

---

## 6. Packaging & CLI

| Command | Purpose |
|---|---|
| `python scripts/download_data.py` | Fetch tabular UCI files |
| `python scripts/download_data.py --with-raw` | + raw IMU signals |
| `python scripts/smoke_test.py` | Fast sanity checks |
| `pytest -q` | Unit tests |
| `python scripts/run_benchmark.py --protocol official` | Classical leaderboard |
| `python scripts/run_benchmark.py --protocol loso` | LOSO leaderboard |
| `python scripts/run_deep_baseline.py` | Features vs CNN |
| `python scripts/export_model.py` | Save cascaded / HGB model |
| `python scripts/predict.py --input FILE` | Batch inference |
| `streamlit run app/streamlit_app.py` | Interactive demo |

Install:

```bash
pip install -r requirements.txt
pip install -e .                # library on PYTHONPATH
pip install -e ".[deep]"        # optional: torch extra (also in requirements)
pip install -e ".[demo]"        # streamlit
```

---

## 7. Python API

```python
from har.data import load_official_splits, ACTIVITY_LABELS
from har.models import build_model_zoo, CascadedHARClassifier
from har.evaluate import evaluate_on_official_test, loso_on_train_partition
from har.knn import CustomKNN
from har.raw_data import load_raw_official_splits
from har.cnn import train_eval_official

# Tabular
train, test = load_official_splits()
zoo = build_model_zoo(include_custom_knn=False)
board = evaluate_on_official_test(zoo)

# Cascade
clf = CascadedHARClassifier().fit(train.X, train.y)
pred = clf.predict(test.X)

# From-scratch KNN
knn = CustomKNN(n_neighbors=5).fit(train.X, train.y)

# Raw CNN
raw_train, raw_test = load_raw_official_splits()
result, metrics, cnn_pred = train_eval_official(epochs=25)
```

---

## 8. Reproducibility

| Knob | Default | Where |
|---|---|---|
| Random seed | `24` | benchmarks, CNN, splits |
| Official split | UCI train/test files | `load_official_splits` |
| CNN epochs | `25` (official), `12` (LOSO) | `run_deep_baseline.py` |
| Batch size | `64` | CNN training |

CI (`.github/workflows/ci.yml`) runs smoke test, pytest (`-m "not slow"`), and a quick official classical benchmark on every push/PR.

Expected drift: different `scikit-learn` / `torch` versions may move scores by a fraction of a percent; ranking should stay stable.

---

## 9. Limitations & future work

### Limitations

1. Waist-only placement  
2. Pre-segmented UCI windows (not a streaming detector)  
3. Sit/stand ambiguity  
4. Random window splits overestimate generalization  

### Future ideas

- Temporal models on raw signals (TCN / Transformer)  
- Streaming / on-device demo (Web Sensor API, mobile)  
- Domain adaptation across device placements  
- Uncertainty-aware UX when sit/stand confidence is low  

---

## 10. References

1. Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN.  
2. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR.  
3. Paszke et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS.  
4. UCI Machine Learning Repository — Human Activity Recognition Using Smartphones Data Set.
