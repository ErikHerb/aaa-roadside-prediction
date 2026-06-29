"""
Regenerate the results charts for the AAA roadside-assistance project.

Reads only aggregate results (model metrics, travel-segment summaries, feature
importances) from the CSV files in this folder. Contains NO member-level data,
identifiers, or personal information -- it is safe to share publicly.

Usage:
    pip install pandas matplotlib
    python plot_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ACCENT = "#1f4e5f"
HIGHLIGHT = "#c0392b"
MUTED = "#7f8c8d"

# ---- Chart 1: model performance comparison ----
models = pd.read_csv("model_results_aaa.csv")
metrics = ["auc", "accuracy", "precision", "recall", "f1"]
labels = ["AUC", "Accuracy", "Precision", "Recall", "F1"]
x = np.arange(len(metrics))
width = 0.26
colors = {"Logistic Regression": HIGHLIGHT, "Random Forest": ACCENT, "Tuned Random Forest": "#2e86ab"}

fig, ax = plt.subplots(figsize=(9, 5))
for j, (_, row) in enumerate(models.iterrows()):
    vals = [row[m] for m in metrics]
    ax.bar(x + (j - 1) * width, vals, width, label=row["model"], color=colors.get(row["model"], MUTED))
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 0.8)
ax.set_ylabel("Score")
ax.set_title("Model performance — roadside-assistance classification")
ax.legend(frameon=False, fontsize=9)
ax.text(0.0, -0.16, "Logistic Regression figures are approximate (read from presentation); others are exact.",
        transform=ax.transAxes, fontsize=8, color=MUTED)
plt.tight_layout()
plt.savefig("model_performance_comparison.png", dpi=150)
plt.close()

# ---- Chart 2: travel segments ----
seg = pd.read_csv("travel_segments_aaa.csv").sort_values("travel_usage_percent")
fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.barh(seg["segment"], seg["travel_usage_percent"], color=ACCENT, edgecolor="white")
for bar, m in zip(bars, seg["members"]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{m:,} members", va="center", fontsize=9, color=MUTED)
ax.set_xlim(0, 11)
ax.set_xlabel("Travel usage rate (%)")
ax.set_title("Travel-focused member segments (K-Means)")
plt.tight_layout()
plt.savefig("travel_segments.png", dpi=150)
plt.close()

# ---- Chart 3: feature importance ----
fi = pd.read_csv("feature_importance_aaa.csv").sort_values("importance")
colors_fi = [HIGHLIGHT if f == "member_tenure_years" else ACCENT for f in fi["feature"]]
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(fi["feature"], fi["importance"], color=colors_fi, edgecolor="white")
ax.set_xlabel("Importance")
ax.set_title("Top features — Tuned Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print("Saved model_performance_comparison.png, travel_segments.png, feature_importance.png")
