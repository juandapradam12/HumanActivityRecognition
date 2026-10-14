# Walkthrough — from clone to leaderboard

A guided path through this repository. Follow it top to bottom the first time; jump sections later.

---

## 0. What you will learn

1. How smartphone IMU windows become activity labels  
2. Why random train/test splits can lie  
3. How a from-scratch k-NN matches scikit-learn  
4. How classical models and a raw-IMU CNN compare under honest protocols  

Time: **~15 minutes** for setup + classical official test; **longer** for full LOSO + CNN.

---

## 1. Install

```bash
git clone https://github.com/juandapradam12/HumanActivityRecognition.git
cd HumanActivityRecognition
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py
```

Check:

```bash
python scripts/smoke_test.py
# expect: smoke_test OK
```

---

## 2. Meet the data (5 minutes)

Open the notebook:

```bash
jupyter notebook "Human Activity Recognition.ipynb"
```

Or in Python:

```python
from har.data import load_official_splits, ACTIVITY_LABELS

train, test = load_official_splits()
print(train.X.shape, test.X.shape)          # (7352, 561), (2947, 561)
print(train.subjects.nunique(), test.subjects.nunique())  # 21, 9
print(ACTIVITY_LABELS)
```

Look at class counts and a correlation heatmap of the first features (`har.plots.plot_feature_correlation`). Gravity-related columns will matter later.

---

## 3. Homemade k-NN

In the notebook (or library):

```python
from har.knn import CustomKNN
from sklearn.neighbors import KNeighborsClassifier

knn = CustomKNN(5).fit(train.X, train.y)
sk = KNeighborsClassifier(5).fit(train.X, train.y)

a = knn.predict(test.X.iloc[:200])
b = sk.predict(test.X.iloc[:200])
print((a == b).mean())  # → 1.0
```

You now have a transparent baseline (~88% on the official test set).

---

## 4. Classical leaderboard (official test)

```bash
python scripts/run_benchmark.py --protocol official --skip-custom-knn
```

You should see logistic regression near **95.6%**, with the cascaded sit/stand model close behind and the **lowest sit↔stand error**.

Optional — subject-independent CV (slower):

```bash
python scripts/run_benchmark.py --protocol loso --skip-custom-knn
```

Expect ~93–94% overall and a noticeable per-subject standard deviation.

---

## 5. Features vs deep (optional but recommended)

```bash
python scripts/download_data.py --with-raw   # ~182MB once
python scripts/run_deep_baseline.py --skip-loso   # official only
# or full:
python scripts/run_deep_baseline.py
```

Compare the printed side-by-side table with `docs/assets/features_vs_cnn.png`.

**Story to remember:** a small 1D CNN on raw waveforms nearly matches carefully engineered 561-D features — but still loses more on sitting vs standing.

---

## 6. Ship something demable

```bash
python scripts/export_model.py
python scripts/predict.py --input X_test.txt | head
streamlit run app/streamlit_app.py
```

In the Streamlit app:

- Browse labeled official-test samples  
- Upload your own whitespace-delimited 561-column windows  

---

## 7. Where the code lives

| You want to… | Open |
|---|---|
| Change data loading | `src/har/data.py`, `raw_data.py` |
| Tweak models / cascade | `src/har/models.py` |
| Change CNN architecture | `src/har/cnn.py` |
| Add a metric / protocol | `src/har/evaluate.py` |
| Adjust CLI experiments | `scripts/run_benchmark.py`, `run_deep_baseline.py` |
| Extend tests | `tests/` |

---

## 8. How to talk about this project

Good one-liner:

> “I built a smartphone HAR stack on UCI data with a from-scratch k-NN, a classical model zoo, and a raw-IMU CNN — evaluated with official held-out subjects and leave-one-subject-out, hitting ~95.6% / ~93.7% respectively.”

Avoid:

> “I got 99% accuracy.”  
(Unless you immediately explain that was a leaked random window split.)

---

## 9. Next experiments you can try

1. Replace the CNN with a TCN or tiny Transformer  
2. Add temperature scaling and a “low confidence” UI state for sit/stand  
3. Collect a few minutes of your own phone IMU and test domain shift  
4. Train only on locomotion vs only on static activities and ensemble them  

Happy hacking.
