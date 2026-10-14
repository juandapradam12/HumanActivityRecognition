# Documentation — Human Activity Recognition

This document describes the dataset, the algorithms used in this repository, how evaluation is performed, and where accuracy still leaves room to grow.

---

## 1. Problem statement

Given a fixed-length window of smartphone inertial measurements, predict one of six activities:

1. Walking  
2. Walking upstairs  
3. Walking downstairs  
4. Sitting  
5. Standing  
6. Laying  

Inputs are **not** raw time series. The UCI authors already transform each 2.56-second window into a **561-dimensional feature vector** of time- and frequency-domain statistics (means, standard deviations, energy, entropy, autoregressive coefficients, signal magnitude area, angle features, etc.).

That makes the task a classic multi-class classification problem on tabular features — ideal for comparing distance-based, linear, kernel, and tree methods.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | [UCI HAR Using Smartphones](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) |
| Volunteers | 30 adults (ages 19–48) |
| Device | Samsung Galaxy S II on the waist |
| Sensors | 3-axis accelerometer + 3-axis gyroscope @ 50 Hz |
| Window | 2.56 s, 50% overlap (128 samples) |
| Features | 561 engineered variables (`features.txt`) |
| Samples in this repo | 7,352 labeled training windows (`X_train.txt`, `y_train.txt`) |

Pre-processing by the dataset authors includes noise filtering and separation of body acceleration from gravity via a Butterworth low-pass filter (0.3 Hz cutoff for gravity).

### Class balance (this repository)

| Label | Activity | Count |
|---:|---|---:|
| 1 | WALKING | 1,226 |
| 2 | WALKING_UPSTAIRS | 1,073 |
| 3 | WALKING_DOWNSTAIRS | 986 |
| 4 | SITTING | 1,286 |
| 5 | STANDING | 1,374 |
| 6 | LAYING | 1,407 |

Classes are reasonably balanced; stratification is still used so hold-out folds preserve the mix.

### Feature naming caveat

A few UCI feature names are duplicated. `har.data.load_feature_names` appends `_1`, `_2`, … so pandas columns stay unique without dropping information.

---

## 3. Algorithms

### 3.1 From-scratch k-Nearest Neighbors

Implemented in `src/har/knn.py` as `CustomKNN`.

**Idea.** For a query vector \(x\):

1. Compute Euclidean distance to every training sample.  
2. Take the \(k\) nearest neighbors (default \(k=5\)).  
3. Predict the majority label.  
4. On a vote tie, prefer the label of the **closest** neighbor (first among sorted neighbors) — the same convention used in the original notebook.

**Performance trick.** Pairwise squared distances are built with one matrix multiply:

\[
\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2\,a^\top b
\]

Neighbor indices use `np.argpartition` so we never fully sort the distance matrix.

**Validation.** On the HAR hold-out slice, predictions match `sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)` with Euclidean distance (zero disagreements on the tested samples).

### 3.2 Classical model zoo

`har.models.build_model_zoo()` builds:

| Model | Notes |
|---|---|
| Custom KNN / sklearn KNN | Pedagogical baseline; ~96% |
| Distance-weighted KNN | Softens hard majority votes |
| Logistic Regression | Strong linear baseline after scaling |
| Linear SVM | Margin-based linear separator |
| RBF SVM | Non-linear kernel; excellent on this feature space |
| PCA(100) + RBF SVM | Compresses correlated IMU features first |
| Random Forest / Extra Trees | Feature-importance insight + robust accuracy |
| Hist Gradient Boosting | Best accuracy in this repo (~99.3%) |

All linear / kernel models are wrapped in a `StandardScaler` pipeline. Tree ensembles operate on the raw (already roughly normalized) UCI features.

### 3.3 Why gravity features dominate

Extra Trees importance consistently ranks:

- `tGravityAcc-energy/mean/min/max` on X/Y  
- `angle(X,gravityMean)`, `angle(Y,gravityMean)`  

That is physically expected:

- **Laying vs upright** is almost entirely an orientation / gravity story.  
- **Walking vs stairs** depends more on body-motion energy and frequency content.  
- **Sitting vs standing** share nearly the same gravity posture at the waist — hence the residual confusion.

---

## 4. Evaluation protocol

Default protocol in the notebook and `scripts/run_benchmark.py`:

```text
train_test_split(X, y, test_size=0.30, random_state=24, stratify=y)
```

Metrics reported:

- Accuracy  
- Weighted F1  
- Macro F1  
- Full classification report + confusion matrix for the best model  

### Important limitation

Windows from the same volunteer are correlated. A random window split **overestimates** subject-independent performance. A stricter next step is leave-one-subject-out (needs the UCI `subject_train.txt` file). This repository currently ships only features and labels, so subject-aware CV is left as a documented extension.

---

## 5. Project layout & API

```text
src/har/
  data.py     load_har_dataset(), ACTIVITY_LABELS
  knn.py      CustomKNN, majority_vote, euclidean_distances
  models.py   build_model_zoo(), evaluate_models(), top_features()
  plots.py    correlation / counts / leaderboard / confusion helpers
scripts/run_benchmark.py
```

Minimal usage:

```python
from har.data import load_har_dataset
from har.knn import CustomKNN
from har.models import build_model_zoo, evaluate_models

X, y = load_har_dataset()
model = CustomKNN(n_neighbors=5).fit(X, y)
```

---

## 6. Reproducing the leaderboard

```bash
pip install -r requirements.txt
python scripts/run_benchmark.py
```

Expected top line (approximate; floating-point / library versions can shift a digit):

```text
Hist Gradient Boosting    accuracy ≈ 0.9937
```

---

## 7. Fixes relative to the original notebook

The first version of this project was a strong teaching notebook with a few sharp edges. This refresh addresses them:

| Issue | Fix |
|---|---|
| `har_train` referenced but never defined | Correlation plot now uses `X` / `plot_feature_correlation` |
| Deprecated `pandas.read_table(..., squeeze=True, sep='\\n')` | `read_csv` with explicit separators |
| Deprecated `np.bool` | Boolean NumPy dtype via `dtype=bool` |
| Slow row-wise homemade KNN | Vectorized `CustomKNN` |
| Only KNN evaluated | Full model zoo + CLI leaderboard |
| Empty README | Project narrative, results table, quick start |
| Duplicate UCI feature names | Deterministic uniquification |

---

## 8. Roadmap

Ideas worth pursuing next:

1. **Subject-independent evaluation** — download `subject_train.txt` / official test split; report leave-one-subject-out scores.  
2. **Raw-signal deep learning** — 1D CNN / LSTM / Transformer on the inertial time series instead of handcrafted features.  
3. **Calibration & latency** — measure prediction time for on-device constraints; distill the boosting model.  
4. **Binary posture detector** — specialize a model for sitting vs standing, the only pair that still confuses the ensemble.  
5. **Streaming demo** — read phone IMU over BLE / Web Sensor API and classify live windows.

---

## 9. References

1. Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN.  
2. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR.  
3. UCI ML Repository — Human Activity Recognition Using Smartphones Data Set.
