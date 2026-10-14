# Model Card — Smartphone Human Activity Recognition

Following the spirit of [Mitchell et al., 2019](https://arxiv.org/abs/1810.03993), this card documents the models shipped in this repository.

---

## Model details

| Field | Value |
|---|---|
| Name | HAR classical zoo + Cascaded HGB/Sit-Stand + optional 1D CNN |
| Version | 1.3.x (see `pyproject.toml` / `har.__version__`) |
| Task | 6-class human activity recognition |
| Frameworks | scikit-learn, PyTorch |
| Author | Juan David Prada Malagón |
| License | Educational / research use (see README) |

### Primary exported artifact

`scripts/export_model.py` saves **`CascadedHARClassifier`** by default:

- Base: `HistGradientBoostingClassifier`  
- Specialist: calibrated RBF SVM on sitting vs standing  
- Serialization: `joblib` + JSON sidecar metadata  

CNN weights from `scripts/run_deep_baseline.py` are saved separately as `har_cnn1d.pt` (not the default export).

---

## Intended use

**Intended**

- Research / education on HAR and evaluation methodology  
- Offline classification of UCI-style 2.56 s waist-IMU windows  
- Benchmarking handcrafted features vs a compact 1D CNN  

**Out of scope**

- Medical diagnosis or fall detection as a clinical device  
- Real-time safety-critical monitoring without further validation  
- Phones in pocket, hand, bag, or different hardware without recalibration  
- Identifying *who* is moving (this is activity classification, not biometrics)

---

## Training data

- **Dataset:** UCI HAR Using Smartphones (Anguita et al., 2013)  
- **Population:** 30 volunteers, ages 19–48  
- **Train subjects:** 21 (7,352 windows)  
- **Test subjects:** 9 (2,947 windows)  
- **Sensors:** waist-mounted Galaxy S II, accel + gyro @ 50 Hz  

Raw inertial channels (optional) are the nine UCI *Inertial Signals* streams, each length 128.

---

## Evaluation data & metrics

| Protocol | Description |
|---|---|
| Official test | Train on 21 subjects → score on 9 held-out subjects |
| LOSO | Leave-one-subject-out on the 21 training subjects |

| Metric | Meaning |
|---|---|
| Accuracy | Overall correct windows |
| Weighted / macro F1 | Class-aware quality |
| Sit↔stand error | Confusion rate within the hard static pair |

### Headline scores (default seeds)

| Model | Official acc. | LOSO acc. |
|---|---:|---:|
| Logistic Regression (features) | 0.9555 | 0.9365 |
| Cascaded HGB + Sit/Stand | 0.9518 | — |
| 1D CNN (raw IMU) | 0.9464 | 0.9278 |

See [`DOCUMENTATION.md`](DOCUMENTATION.md#5-reported-results) for the full tables.

---

## Ethical considerations

- Data were collected in a controlled lab setting with adult volunteers; performance may not transfer to children, elderly, or clinical populations.  
- Activity labels can be privacy-sensitive in some contexts (inferring lifestyle). Deploy with consent and clear purpose limitation.  
- Do not present random hold-out accuracy as real-world performance.  

---

## Limitations & failure modes

1. **Sitting vs standing** — dominant residual error (waist IMU looks similar when still).  
2. **Placement shift** — models assume waist mounting.  
3. **Device / sampling drift** — different phones or Hz may degrade scores.  
4. **Segmented input** — expects pre-cut windows, not continuous stream post-processing.  
5. **CNN variance** — LOSO subject std ≈ 0.09; some subjects are much harder.

---

## How to use responsibly

```bash
# Prefer subject-aware metrics when reporting
python scripts/run_benchmark.py --protocol official --skip-custom-knn
python scripts/run_benchmark.py --protocol loso --skip-custom-knn

# Export + predict on UCI-format feature rows
python scripts/export_model.py
python scripts/predict.py --input your_windows.txt
```

When confidence on sitting/standing matters, use the calibrated specialist probabilities (`CascadedHARClassifier.predict_proba_specialist`) and surface low-confidence cases to a human or secondary signal.
