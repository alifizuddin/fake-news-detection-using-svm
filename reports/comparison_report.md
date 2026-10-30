# Fake News Detection - Model Comparison Report

## Dataset audit

- Shape after cleaning: 21601
- Training samples: 17282
- Test samples: 4319
- Split method: StratifiedGroupKFold (group key: normalized title)
- Class balance (train): {'0': 4137, '1': 13145}
- Class balance (test): {'0': 1063, '1': 3256}

## Baselines
- Majority class (most_frequent): acc=0.7539, macroF1=0.4298, fakeF1=0.0000, realF1=0.8597
- Stratified: acc=0.6365, macroF1=0.5035, fakeF1=0.2466, realF1=0.7605

## Standard classifiers
- Multinomial Naive Bayes (TF-IDF 1-2): acc=0.8400, macroF1=0.7497, fakeF1=0.5994, realF1=0.9000
- Logistic Regression (TF-IDF 1-2, balanced): acc=0.8120, macroF1=0.7643, fakeF1=0.6582, realF1=0.8703

## Linear SVM (tuned with CV on training data only)
- Best params: {'svm__C': 0.1, 'tfidf__max_features': 5000, 'tfidf__min_df': 2, 'tfidf__ngram_range': (1, 1)}
- Best CV macro F1: 0.7567
- Test set: acc=0.8111, macroF1=0.7614, fakeF1=0.6525, realF1=0.8703

## Ablation (unigram vs unigram+bigram, LinearSVC)
- Unigram (1,1): acc=0.7930, macroF1=0.7422, fakeF1=0.6278, realF1=0.8566
- Unigram+bigram (1,2): acc=0.7960, macroF1=0.7450, fakeF1=0.6309, realF1=0.8591

## Conclusion
The linear SVM remains competitive; tuning shows the selected configuration. Adding bigrams
is tested via ablation to assess added complexity.
