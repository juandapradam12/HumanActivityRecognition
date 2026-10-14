"""Interactive demo for Human Activity Recognition predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from har.data import ACTIVITY_LABELS, load_official_splits
from har.export import load_model, save_model
from har.models import CascadedHARClassifier

MODEL_PATH = ROOT / "artifacts" / "models" / "har_cascaded.joblib"


@st.cache_resource
def get_model():
    if MODEL_PATH.exists():
        return load_model(MODEL_PATH), "exported model"

    train, _ = load_official_splits(with_subjects=False)
    model = CascadedHARClassifier(random_state=24)
    model.fit(train.X, train.y)
    save_model(
        model,
        MODEL_PATH,
        meta={"model_name": "Cascaded HGB + Sit/Stand", "source": "streamlit"},
    )
    return model, "trained on demand"


@st.cache_data
def load_demo_rows(n: int = 12):
    _, test = load_official_splits(with_subjects=True)
    # Stratified-ish sample: one or two rows per class
    rows = []
    for label in sorted(ACTIVITY_LABELS):
        idx = test.y[test.y == label].index[:2]
        rows.extend(list(idx))
    chosen = rows[:n]
    return test.X.loc[chosen].reset_index(drop=True), test.y.loc[chosen].reset_index(
        drop=True
    ), test.subjects.loc[chosen].reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="HAR Demo", page_icon="📡", layout="wide")
    st.title("Human Activity Recognition")
    st.caption(
        "Classify smartphone IMU windows into walking, stairs, sitting, standing, or laying."
    )

    model, source = get_model()
    st.sidebar.success(f"Model ready ({source})")
    st.sidebar.markdown(
        """
**Activities**
1. Walking  
2. Walking upstairs  
3. Walking downstairs  
4. Sitting  
5. Standing  
6. Laying  
"""
    )

    tab_sample, tab_upload = st.tabs(["Official test samples", "Upload features"])

    with tab_sample:
        X, y, subjects = load_demo_rows()
        idx = st.slider("Sample index", 0, len(X) - 1, 0)
        row = X.iloc[[idx]]
        true_label = int(y.iloc[idx])
        pred = int(model.predict(row)[0])

        c1, c2, c3 = st.columns(3)
        c1.metric("True activity", ACTIVITY_LABELS[true_label])
        c2.metric("Predicted", ACTIVITY_LABELS[pred])
        c3.metric("Subject", int(subjects.iloc[idx]))

        st.write("Selected feature window (first 20 of 561 dimensions)")
        st.line_chart(row.iloc[0, :20])
        st.dataframe(row.T.rename(columns={0: "value"}).head(30))

    with tab_upload:
        st.write(
            "Upload a whitespace-delimited `.txt`/`.csv` with **561 columns** "
            "(one window per row), matching UCI HAR feature order."
        )
        uploaded = st.file_uploader("Feature matrix", type=["txt", "csv"])
        if uploaded is not None:
            data = pd.read_csv(uploaded, sep=r"\s+", header=None)
            if data.shape[1] != 561:
                st.error(f"Expected 561 features, got {data.shape[1]}")
            else:
                preds = model.predict(data)
                out = pd.DataFrame(
                    {
                        "label": preds,
                        "activity": [ACTIVITY_LABELS[int(p)] for p in preds],
                    }
                )
                st.dataframe(out)
                st.bar_chart(out["activity"].value_counts())


if __name__ == "__main__":
    main()
