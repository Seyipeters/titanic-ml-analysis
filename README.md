# Titanic - ML Analysis

> Entry-level AI/ML portfolio project: end-to-end machine learning workflow.

**Dataset**: titanic | **Source**: seaborn built-in | **Rows**: 891 | **Target**: `survived`

## Key Insights

- Dataset contains 891 rows and 15 columns.
- Target 'survived' has 2 classes: ['0', '1'].
- Model achieved 100.0% test accuracy.
- 5-fold CV accuracy: 100.0% +/- 0.0%.
- Top numeric features used: pclass, age, sibsp, parch.

## Model Performance (RandomForestClassifier)

| Metric | Value |
|--------|-------|
| accuracy | `1.0` |
| cv_mean_accuracy | `1.0` |
| cv_std | `0.0` |
| train_samples | `712` |
| test_samples | `179` |
| n_classes | `2` |

## Tech Stack

- Python 3.10+, pandas, NumPy, scikit-learn (`RandomForestClassifier`), matplotlib, seaborn

## How to Run

```bash
pip install pandas scikit-learn matplotlib seaborn
python 01_data_exploration.py
python 02_preprocessing.py
python 03_model_training.py
python 04_results.py
```
