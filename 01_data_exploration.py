"""Stage 1: Data Exploration"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

df = pd.read_csv("../data/titanic.csv")
print("Shape:", df.shape)
print(df.isnull().sum())
print(df.describe())
num_cols = df.select_dtypes(include="number").columns.tolist()
Path("plots").mkdir(exist_ok=True)
if num_cols:
    df[num_cols[:6]].hist(bins=20, figsize=(12, 8))
    plt.suptitle("titanic - Distributions")
    plt.tight_layout()
    plt.savefig("plots/distributions.png", dpi=120)
if len(num_cols) >= 2:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("titanic - Correlation")
    plt.tight_layout()
    plt.savefig("plots/correlation.png", dpi=120)
print("EDA complete.")
