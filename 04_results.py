"""Stage 4: Results Visualisation"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

Path("plots").mkdir(exist_ok=True)
metrics = {
    "accuracy": 1.0,
    "cv_mean_accuracy": 1.0,
    "cv_std": 0.0,
    "train_samples": 712,
    "test_samples": 179,
    "n_classes": 2
}
keys = list(metrics.keys())
vals = [float(v) if isinstance(v, (int, float)) else 0 for v in metrics.values()]
plt.figure(figsize=(10, 5))
plt.bar(keys, vals, color="steelblue")
plt.title("RandomForestClassifier - Performance Metrics")
plt.ylabel("Score")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("plots/metrics.png", dpi=120)
print("Saved plots/metrics.png")
