# Titanic - ML Analysis

> Entry-level AI/ML portfolio project: end-to-end machine learning workflow.

## 🎯 Project Objectives

This project demonstrates:
- **Data Exploration**: Comprehensive EDA with distributions, correlations, and missing value analysis
- **Feature Engineering**: Numeric scaling, categorical encoding, and preprocessing
- **Model Training**: Cross-validated RandomForestClassifier with performance benchmarking
- **Visualization**: Minimalist charts (doughnut pies, distributions, heatmaps)

## 🎯 Project Aim

Build a reproducible ML pipeline for **titanic** dataset that:
1. Loads and explores 891 rows of data
2. Engineers features with 2 validation strategies
3. Trains RandomForestClassifier for survived prediction
4. Evaluates with cross-validation and test metrics
5. Generates interactive Jupyter notebook for reproducibility

## 📊 Dataset Overview

| Property | Value |
|----------|-------|
| Dataset | titanic |
| Source | seaborn built-in |
| Rows | 891 |
| Target | `survived` |

## 📈 Model Performance (RandomForestClassifier)

| Metric | Value |
|--------|-------|
| accuracy | `1.0` |
| cv_mean_accuracy | `1.0` |
| cv_std | `0.0` |
| train_samples | `712` |
| test_samples | `179` |
| n_classes | `2` |

## 💡 Key Insights

- Dataset contains 891 rows and 15 columns.
- Target 'survived' has 2 classes: ['0', '1'].
- Model achieved 100.0% test accuracy.
- 5-fold CV accuracy: 100.0% +/- 0.0%.
- Top numeric features used: pclass, age, sibsp, parch.

## 📊 Visualizations

The interactive Jupyter notebook (`analysis.ipynb`) includes:
- **Numeric Distributions**: Histograms of all numeric features
- **Categorical Doughnuts**: Minimalist doughnut pie charts for category breakdowns
- **Correlation Heatmap**: Pairwise feature correlations with coolwarm colormap
- **Metrics Dashboard**: Bar chart + doughnut visualization of model performance

## 🚀 How to Use

### Option 1: Jupyter Notebook (Interactive)
```bash
pip install jupyter pandas scikit-learn matplotlib seaborn
jupyter notebook analysis.ipynb
```

### Option 2: Command Line
```bash
python -c "from analysis import run; run()"
```

## 🛠️ Tech Stack

- **Python 3.9+** - Core language
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning (RandomForestClassifier)
- **matplotlib** - Static visualizations
- **seaborn** - Statistical graphics
- **Jupyter** - Interactive notebook

## 📝 License

Open source - Free to use for portfolio and educational purposes.
