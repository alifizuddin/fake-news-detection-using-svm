import json
from pathlib import Path

import pandas as pd
import re

import nltk
from nltk.corpus import stopwords

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
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
    audit["duplicate_title_rows"] = duplicate_title_rows

    conflict_titles = (
        df.groupby("title")["real"].nunique().gt(1).sum()
        if "title" in df.columns
        else 0
    )
    audit["titles_with_conflicting_labels"] = int(conflict_titles)

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
    print(f"Titles with conflicting labels: {conflict_titles}")
    print("=" * 70 + "\n")

    return audit


def load_and_clean_data(path=DATASET_PATH):
    df = pd.read_csv(path)
    audit = audit_dataset(df)

    required = {"title", "real"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    df = df.copy()
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"].str.len() > 0]
    df = df[df["real"].isin([0, 1])]

    conflict_titles = df.groupby("title")["real"].nunique()
    conflict_titles = conflict_titles[conflict_titles > 1].index
    df = df[~df["title"].isin(conflict_titles)]
    df = df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

    return df, audit


def make_group_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    groups = df["title"].str.lower().str.strip()
    y = df["real"].astype(int)

    n_splits = int(round(1 / test_size))
    n_splits = max(2, n_splits)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(df["title"], y, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def compute_metrics(y_true, y_pred, y_scores=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(
            precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        ),
        "recall_fake": float(
            recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        ),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_real": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_real": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_real": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_scores is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def run_comparisons(train_df, X_train_text, X_test_text, y_train, y_test):
    results = {}

    # Baselines
    for strategy in ["most_frequent", "stratified"]:
        dummy = DummyClassifier(strategy=strategy, random_state=RANDOM_STATE)
        dummy.fit(X_train_text, y_train)
        y_pred = dummy.predict(X_test_text)
        y_scores = None
        if hasattr(dummy, "predict_proba"):
            try:
                y_scores = dummy.predict_proba(X_test_text)[:, 1]
            except Exception:
                pass
        elif hasattr(dummy, "decision_function"):
            try:
                y_scores = dummy.decision_function(X_test_text)
            except Exception:
                pass
        results[f"dummy_{strategy}"] = compute_metrics(y_test, y_pred, y_scores)

    # Standard text classifiers (default params)
    nb_pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2,
                    stop_words="english",
                ),
            ),
            ("nb", MultinomialNB()),
        ]
    )
    nb_pipe.fit(X_train_text, y_train)
    results["multinomial_nb"] = compute_metrics(
        y_test, nb_pipe.predict(X_test_text)
    )

    lr_pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2,
                    stop_words="english",
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    lr_pipe.fit(X_train_text, y_train)
    y_scores_lr = None
    try:
        y_scores_lr = lr_pipe.decision_function(X_test_text)
    except Exception:
        try:
            y_scores_lr = lr_pipe.predict_proba(X_test_text)[:, 1]
        except Exception:
            pass
    results["logistic_regression"] = compute_metrics(
        y_test, lr_pipe.predict(X_test_text), y_scores_lr
    )

    # SVM with CV tuning
    svm_pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            (
                "svm",
                LinearSVC(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=10000,
                ),
            ),
        ]
    )
    param_grid = {
        "tfidf__max_features": [3000, 5000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 5],
        "svm__C": [0.1, 1.0],
    }
    groups_train = train_df["title"].str.lower().str.strip()
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    search = GridSearchCV(
        svm_pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_train_text, y_train, groups=groups_train)
    best_svm = search.best_estimator_
    y_pred_best = best_svm.predict(X_test_text)
    y_scores_best = None
    try:
        y_scores_best = best_svm.decision_function(X_test_text)
    except Exception:
        pass
    results["svm_linear_tuned"] = compute_metrics(y_test, y_pred_best, y_scores_best)
    results["svm_tuning"] = {
        "best_params": search.best_params_,
        "best_cv_score_macro_f1": float(search.best_score_),
    }

    # Ablation: unigram vs unigram+bigram
    ablation = {}
    for ngram in [(1, 1), (1, 2)]:
        ab_pipe = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=5000,
                        ngram_range=ngram,
                        min_df=2,
                        stop_words="english",
                    ),
                ),
                (
                    "svm",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        max_iter=10000,
                    ),
                ),
            ]
        )
        ab_pipe.fit(X_train_text, y_train)
        y_pred_ab = ab_pipe.predict(X_test_text)
        ablation[f"ngram_{ngram}"] = compute_metrics(y_test, y_pred_ab)
    results["ablation"] = ablation

    return results, best_svm


def save_reports(audit, split_info, results):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORTS_DIR / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    with open(REPORTS_DIR / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    with open(REPORTS_DIR / "comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Create summary report
    def fmt(m):
        return (
            f"acc={m['accuracy']:.4f}, macroF1={m['macro_f1']:.4f}, "
            f"fakeF1={m['f1_fake']:.4f}, realF1={m['f1_real']:.4f}"
        )

    report = f"""# Fake News Detection - Model Comparison Report

## Dataset audit

- Shape after cleaning: {split_info['train_size'] + split_info['test_size']}
- Training samples: {split_info['train_size']}
- Test samples: {split_info['test_size']}
- Split method: {split_info['split_method']}
- Class balance (train): {split_info['train_class_balance']}
- Class balance (test): {split_info['test_class_balance']}

## Baselines
- Majority class (most_frequent): {fmt(results['dummy_most_frequent'])}
- Stratified: {fmt(results['dummy_stratified'])}

## Standard classifiers
- Multinomial Naive Bayes (TF-IDF 1-2): {fmt(results['multinomial_nb'])}
- Logistic Regression (TF-IDF 1-2, balanced): {fmt(results['logistic_regression'])}

## Linear SVM (tuned with CV on training data only)
- Best params: {results['svm_tuning']['best_params']}
- Best CV macro F1: {results['svm_tuning']['best_cv_score_macro_f1']:.4f}
- Test set: {fmt(results['svm_linear_tuned'])}

## Ablation (unigram vs unigram+bigram, LinearSVC)
- Unigram (1,1): {fmt(results['ablation']['ngram_(1, 1)'])}
- Unigram+bigram (1,2): {fmt(results['ablation']['ngram_(1, 2)'])}

## Conclusion
The linear SVM remains competitive; tuning shows the selected configuration. Adding bigrams
is tested via ablation to assess added complexity.
"""
    with open(REPORTS_DIR / "comparison_report.md", "w", encoding="utf-8") as f:
        f.write(report)


def predict_news(pipeline, title):
    title_clean = clean_text(title)
    result = pipeline.predict([title_clean])
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
            str(k): int(v)
            for k, v in y_train.value_counts().sort_index().items()
        },
        "test_class_balance": {
            str(k): int(v)
            for k, v in y_test.value_counts().sort_index().items()
        },
    }

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Train class balance: {split_info['train_class_balance']}")
    print(f"Test class balance: {split_info['test_class_balance']}\n")

    print("Running comparisons and tuning (CV on training data only)...")
    results, best_svm = run_comparisons(train_df, X_train_text, X_test_text, y_train, y_test)

    print("\n=== Baselines ===")
    print("majority:", results["dummy_most_frequent"])
    print("stratified:", results["dummy_stratified"])
    print("\n=== Classifiers ===")
    print("NB:", results["multinomial_nb"])
    print("LR:", results["logistic_regression"])
    print("\n=== SVM tuned ===")
    print("best params:", results["svm_tuning"]["best_params"])
    print("best CV macroF1:", results["svm_tuning"]["best_cv_score_macro_f1"])
    print("test:", results["svm_linear_tuned"])
    print("\n=== Ablation ===")
    for k, v in results["ablation"].items():
        print(k, v)

    save_reports(audit, split_info, results)
    print(f"\nReports saved to: {REPORTS_DIR}")

    run_gui(best_svm)


if __name__ == "__main__":
    main()
