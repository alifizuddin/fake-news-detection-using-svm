# Fake News Detection - Evaluation Report

## Dataset audit

- Shape after cleaning: 21601
- Training samples: 17282
- Test samples: 4319
- Split method: StratifiedGroupKFold (group key: normalized title)
- Class balance (train): {'0': 4137, '1': 13145}
- Class balance (test): {'0': 1063, '1': 3256}

## Model

- Vectorizer: TF-IDF (max_features=5000, ngram_range=(1, 2), min_df=2)
- Classifier: SVM with linear kernel and balanced class weights
- Pipeline: TfidfVectorizer -> SVC

## Results

- Accuracy: 0.7960
- Macro F1: 0.7450
- Fake precision/recall/F1: 0.5687 / 0.7084 / 0.6309
- Real precision/recall/F1: 0.8965 / 0.8246 / 0.8591
- ROC-AUC: 0.8472105047834116

## Confusion matrix

Rows are true labels and columns are predicted labels (Fake, Real):

```
[[ 753  310]
 [ 571 2685]]
```
