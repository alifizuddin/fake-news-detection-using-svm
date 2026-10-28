import json
from pathlib import Path

import numpy as np
import pandas as pd
import re

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import tkinter as tk
from tkinter import messagebox

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "FakeNewsNet.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_stopwords():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


ensure_stopwords()
STOP_WORDS = set(stopwords.words("english"))


def clean_text(text):
    """Lowercase, remove URLs/punctuation, and remove English stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|[^a-z\s]", " ", text)
    text = " ".join(word for word in text.split() if word not in STOP_WORDS)
    return text


def audit_dataset(df):
    """Print and return an audit of labels, balance, missing values, and duplicates."""
    audit = {}

    audit["shape"] = list(df.shape)
    audit["columns"] = list(df.columns)
    audit["missing_values"] = {k: int(v) for k, v in df.isna().sum().items()}

    label_counts = df["real"].value_counts().sort_index().to_dict()
    audit["label_counts"] = {str(k): int(v) for k, v in label_counts.items()}
    total = len(df)
    audit["class_balance"] = {
        str(k): round(int(v) / total, 4) for k, v in label_counts.items()
    }

    duplicate_title_rows = int(df["title"].duplicated().sum())
    duplicate_unique_titles = int(df["title"].duplicated(keep=False).sum() > 0 and
                                 df.loc[df["title"].duplicated(keep=False), "title"].nunique())
    audit["duplicate_title_rows"] = duplicate_title_rows
    audit["duplicate_unique_titles"] = duplicate_unique_titles

    conflicting = (
        df.groupby("title")["real"].nunique().gt(1).sum()
        if "title" in df.columns and "real" in df.columns
        else 0
    )
    audit["titles_with_conflicting_labels"] = int(conflicting)

    print("\n" + "=" * 70)
    print("DATASET AUDIT")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nMissing values:")
    for col, count in audit["missing_values"].items():
        print(f"  {col}: {count}")
    print("\nLabel counts (0 = fake, 1 = real):")
    for label, count in audit["label_counts"].items():
        print(f"  {label}: {count} ({audit['class_balance'][label]:.2%})")
    print(f"\nDuplicate title rows: {duplicate_title_rows}")
    print(f"Unique titles appearing more than once: {duplicate_unique_titles}")
    print(f"Titles with conflicting labels: {conflicting}")
    print("=" * 70 + "\n")

    return audit


def load_and_clean_data(path=DATASET_PATH):
    df = pd.read_csv(path)
    audit = audit_dataset(df)

    # Basic validation.
    required = {"title", "real"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    df = df.copy()
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"].str.len() > 0]

    # Remove rows with missing labels.
    df = df[df["real"].isin([0, 1])]

    # Resolve duplicate titles: drop titles whose labels conflict, then keep one row.
    conflict_titles = df.groupby("title")["real"].nunique()
    conflict_titles = conflict_titles[conflict_titles > 1].index
    df = df[~df["title"].isin(conflict_titles)]
    df = df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

    return df, audit


def make_group_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Stratified, group-aware holdout split using title as the group key."""
    groups = df["title"].str.lower().str.strip()
    y = df["real"].astype(int)

    # StratifiedGroupKFold gives a stratified split while keeping duplicate titles
    # (groups) out of both train and test sets.
    n_splits = int(round(1 / test_size))
    n_splits = max(2, n_splits)
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    train_idx, test_idx = next(sgkf.split(df["title"], y, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def build_pipeline():
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    min_df=2,
                ),
            ),
            (
                "svm",
                LinearSVC(
                    class_weight="balanced",
                    C=1.0,
                    random_state=RANDOM_STATE,
                    max_iter=10000,
                ),
            ),
        ]
    )


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_fake": float(precision_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "recall_fake": float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "f1_fake": float(f1_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "precision_real": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall_real": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1_real": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # ROC-AUC from SVM decision scores.
    try:
        scores = pipeline.decision_function(X_test)
        metrics["roc_auc"] = float(roc_auc_score(y_test, scores))
    except Exception:
        metrics["roc_auc"] = None

    print("Accuracy:", metrics["accuracy"])
    print("Macro F1:", metrics["macro_f1"])
    print("Fake  - precision/recall/F1:",
          metrics["precision_fake"], metrics["recall_fake"], metrics["f1_fake"])
    print("Real  - precision/recall/F1:",
          metrics["precision_real"], metrics["recall_real"], metrics["f1_real"])
    if metrics["roc_auc"] is not None:
        print("ROC-AUC:", metrics["roc_auc"])
    print("\nConfusion matrix [rows=true, cols=pred; order Fake, Real]:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    return metrics, y_pred


def save_reports(audit, split_info, metrics):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORTS_DIR / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    with open(REPORTS_DIR / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = f"""# Fake News Detection - Evaluation Report

## Dataset audit

- Shape after cleaning: {split_info['train_size'] + split_info['test_size']}
- Training samples: {split_info['train_size']}
- Test samples: {split_info['test_size']}
- Split method: {split_info['split_method']}
- Class balance (train): {split_info['train_class_balance']}
- Class balance (test): {split_info['test_class_balance']}

## Model

- Vectorizer: TF-IDF (max_features=5000, ngram_range=(1, 2), min_df=2)
- Classifier: SVM with linear kernel and balanced class weights
- Pipeline: TfidfVectorizer -> SVC

## Results

- Accuracy: {metrics['accuracy']:.4f}
- Macro F1: {metrics['macro_f1']:.4f}
- Fake precision/recall/F1: {metrics['precision_fake']:.4f} / {metrics['recall_fake']:.4f} / {metrics['f1_fake']:.4f}
- Real precision/recall/F1: {metrics['precision_real']:.4f} / {metrics['recall_real']:.4f} / {metrics['f1_real']:.4f}
- ROC-AUC: {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'}

## Confusion matrix

Rows are true labels and columns are predicted labels (Fake, Real):

```
{np.array(metrics['confusion_matrix'])}
```
"""
    with open(REPORTS_DIR / "evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report)


def predict_news(pipeline, title):
    title = clean_text(title)
    result = pipeline.predict([title])
    return "REAL" if result[0] == 1 else "FAKE"


def run_gui(pipeline):
    window = tk.Tk()
    window.title("Fake News Detector")

    tk.Label(window, text="Enter News Title:", font=("Arial", 12)).pack(pady=10)
    entry = tk.Entry(window, width=60)
    entry.pack(pady=5)

    def on_check():
        title = entry.get()
        if not title.strip():
            messagebox.showwarning("Input Error", "Please enter a news title.")
            return
        result = predict_news(pipeline, title)
        messagebox.showinfo("Prediction Result", f"This news is likely: {result}")

    tk.Button(window, text="Check", command=on_check).pack(pady=10)
    window.mainloop()


def main():
    df, audit = load_and_clean_data()

    # Group-aware, stratified holdout split (no title leakage across splits).
    train_df, test_df = make_group_split(df)

    X_train_text = train_df["title"].apply(clean_text)
    X_test_text = test_df["title"].apply(clean_text)
    y_train = train_df["real"].astype(int)
    y_test = test_df["real"].astype(int)

    split_info = {
        "split_method": "StratifiedGroupKFold (group key: normalized title)",
        "test_size_target": TEST_SIZE,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "train_class_balance": {
            str(k): int(v) for k, v in y_train.value_counts().sort_index().items()
        },
        "test_class_balance": {
            str(k): int(v) for k, v in y_test.value_counts().sort_index().items()
        },
    }

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Train class balance: {split_info['train_class_balance']}")
    print(f"Test class balance: {split_info['test_class_balance']}\n")

    pipeline = build_pipeline()
    pipeline.fit(X_train_text, y_train)

    metrics, _ = evaluate_model(pipeline, X_test_text, y_test)
    save_reports(audit, split_info, metrics)

    print(f"\nReports saved to: {REPORTS_DIR}")

    run_gui(pipeline)


if __name__ == "__main__":
    main()
