# Human Activity Recognition from Smartphone Sensors

> Turn waist-mounted accelerometer & gyroscope readings into a real-time guess of what someone is doing — walking, climbing stairs, sitting, standing, or lying down — at **~99% accuracy**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20HAR-green)](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

This project classifies human activity from the classic [UCI HAR Using Smartphones](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) dataset: **7,352 labeled windows**, each described by **561 engineered IMU features**.

It is both a **teaching notebook** (k-NN implemented from scratch) and a **serious classical-ML benchmark** that pushes accuracy from a ~96% neighbor baseline to **~99.3%** with histogram gradient boosting.

---

## Why this matters

Smartphones already carry the sensors needed for:

- Fitness & rehab coaching
- Fall / sedentary detection for health apps
- Context-aware UX (silence notifications while walking upstairs, etc.)
- Privacy-friendly on-device sensing — no camera required

HAR is a clean playground for comparing distance-based, linear, kernel, and ensemble methods on a high-dimensional but well-structured feature space.

---

## Results at a glance

Hold-out evaluation: stratified 70/30 split of the training windows (`random_state=24`).

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Hist Gradient Boosting | **0.9937** | **0.9937** |
| RBF SVM | 0.9882 | 0.9882 |
| Extra Trees | 0.9878 | 0.9878 |
| Logistic Regression | 0.9841 | 0.9841 |
| Linear SVM | 0.9819 | 0.9819 |
| Random Forest | 0.9814 | 0.9814 |
| PCA(100) + RBF SVM | 0.9764 | 0.9764 |
| KNN (k=5) | ~0.958 | ~0.958 |
| **Custom KNN (from scratch)** | matches sklearn | — |

Almost every remaining mistake is **sitting ↔ standing** — exactly the pair that looks alike on a waist IMU when the body is still.

---

## What’s inside

```
├── Human Activity Recognition.ipynb   # End-to-end story: EDA → homemade KNN → model zoo
├── src/har/                           # Reusable library
│   ├── data.py                        # Dataset loading & activity labels
│   ├── knn.py                         # Vectorized from-scratch k-NN
│   ├── models.py                      # Model zoo & evaluation
│   └── plots.py                       # EDA / comparison charts
├── scripts/run_benchmark.py           # One-command leaderboard
├── docs/DOCUMENTATION.md              # Methodology, algorithms, design notes
├── features.txt                       # 561 feature names
├── activity_labels.txt                # 1..6 → activity names
├── X_train.txt / y_train.txt          # UCI HAR training partition
└── requirements.txt
```

---

## Quick start

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Rank every model (skip homemade KNN for a faster smoke run)
python scripts/run_benchmark.py
python scripts/run_benchmark.py --skip-custom-knn

# 3. Explore interactively
jupyter notebook "Human Activity Recognition.ipynb"
```

Use the library directly:

```python
from har.data import load_har_dataset
from har.models import build_model_zoo, evaluate_models
from sklearn.model_selection import train_test_split

X, y = load_har_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=24, stratify=y
)
results = evaluate_models(build_model_zoo(), X_train, y_train, X_test, y_test)
print(results.head())
```

---

## Activities

| ID | Activity |
|---:|---|
| 1 | WALKING |
| 2 | WALKING_UPSTAIRS |
| 3 | WALKING_DOWNSTAIRS |
| 4 | SITTING |
| 5 | STANDING |
| 6 | LAYING |

Signals come from a Samsung Galaxy S II worn on the waist (50 Hz accelerometer + gyroscope), filtered and windowed into 2.56 s segments with 50% overlap. See [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) for the full pipeline.

---

## Highlights

- **From-scratch KNN that matches scikit-learn** — Euclidean distances, majority vote, nearest-neighbor tie-break, vectorized with NumPy.
- **Model zoo beyond neighbors** — logistic regression, linear/RBF SVM, PCA compression, Random Forest, Extra Trees, Hist Gradient Boosting.
- **Feature intelligence** — gravity channels and device–gravity angles dominate importance, which matches the physics of posture.
- **Clean package layout** — notebook for narrative, `src/har` for reuse, CLI for repeatable benchmarks.

---

## Documentation

- [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) — dataset, algorithms, evaluation protocol, improvement roadmap
- Notebook walkthrough — open `Human Activity Recognition.ipynb`

---

## License & data credit

Code in this repository is provided for educational and research use. The sensor data belongs to the original UCI HAR authors (Reyes-Ortiz, Anguita, Ghio, Oneto, Parra). Please cite their work if you publish results based on this dataset:

> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz.  
> *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN 2013.
