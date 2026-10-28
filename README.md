✅ Project Checklist: Fake News Detection using SVM

## 🧑‍🤝‍🧑 Group & Proposal

[✅] Form group (4–5 members)

[✅] Assign roles to each member

[✅] Choose a problem: Fake News Detection

[✅] Choose dataset 

[✅] Choose algorithm: Support Vector Machine (SVM)

[✅] Write justification for using SVM

[✅] Prepare proposal

[✅] Submit proposal

## 💻 Develop the Solution

[✅] Load dataset with pandas

[✅] Select required columns (e.g. title, real)

[✅] Preprocess and clean text data

[✅] Convert text to numbers using TF-IDF

[✅] Split data into training and testing sets

[✅] Train model using SVM

[✅] Evaluate model accuracy and classification report

[✅] Try improvements (e.g. ngram_range, stop_words)

[✅] Create prediction function for user input

[✅] Create GUI using tkinter

[✅] Organize code in one file (main.py)

[✅] Add comments and make code readable

## 📘 User Manual

[✅] Write installation steps (e.g. how to run, install libraries)

[✅] Include usage instructions

[✅] Add screenshots of model running or interface

[✅] Save manual as PDF or Word document

## 🔄 Flowchart & Pseudocode

[✅] Draw flowchart showing system workflow

[✅] Write pseudocode to explain logic

[✅] Include both in final report or appendix

## 📝 Final Report

[🔲] Declaration page (signed by members)

[✅] Introduction (explain problem and importance)

[✅] Algorithm selection (justify SVM)

[✅] Development steps (based on user manual)

[✅] Results and discussion (accuracy, findings)

[✅] Conclusion (summary + suggestions)

[✅] References (APA/IEEE style)

## 📄 Research Paper (Springer/IJIC Format)

[✅] Title and group member names

[✅] Abstract

[✅] Background and literature review

[✅] Methodology (detailed explanation)

[✅] Results and discussion

[✅] Conclusion and references

[✅] Use correct formatting and structure

## 🎥 Project Presentation

[✅] Prepare PowerPoint slides (problem, method, result, conclusion)

[✅] Include flowchart or screenshots in slides

[✅] Record video presentation (5–10 mins)

[✅] Make sure all members participate

## Current Implementation Notes

### Setup

```bash
python -m pip install -r requirements.txt
```

NLTK stopwords are downloaded automatically if missing.

### Run

```bash
python main.py
```

The script first audits the dataset, trains a TF-IDF + linear SVM pipeline, evaluates it on a
stratified group-aware holdout split, saves reports to `reports/`, and then opens the GUI.

### Methodology

- Dataset audit: labels, class balance, missing values, duplicate titles, conflicting labels.
- Text cleaning: lowercasing, URL/punctuation removal, stopword removal.
- Splitting: `StratifiedGroupKFold` with the normalized title as the group key, so duplicate
  titles cannot leak across train and test sets.
- Vectorization and classifier are combined in one scikit-learn `Pipeline`, so TF-IDF is fit
  only on training data.
- Classifier: linear SVM (`LinearSVC`) with balanced class weights.

### Latest holdout results

See `reports/evaluation_report.md` and `reports/metrics.json`.

## 📤 Final Submission

[✅] Upload research paper (Springer/IJIC format)

[✅] Upload source code + user manual + flowchart + pseudocode

[🔲] Upload slides and video presentation

[✅] Use correct filename (e.g., AliSaraMohd.pdf/.zip)

## 🧾 Meeting Report

[✅] Write one-page meeting summary and task delegation

[✅] Sign by all group members

[✅] Submit with other documents

