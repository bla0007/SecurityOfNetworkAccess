# %% [markdown]
# # SONA — Security of Network Access
# ## Exploratory Data Analysis
# **Week 1 deliverable**: Understand the NSL-KDD dataset before building models.
#
# Run each cell and study the outputs — this is where you learn what the data looks like.

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, "../src")
from preprocess import COLUMNS, ATTACK_MAP, CATEGORICAL_COLS

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# %% [markdown]
# ## 1. Load data

# %%
train = pd.read_csv("../data/KDDTrain+.txt", header=None, names=COLUMNS)
test  = pd.read_csv("../data/KDDTest+.txt",  header=None, names=COLUMNS)

print(f"Train: {train.shape[0]:,} rows × {train.shape[1]} columns")
print(f"Test : {test.shape[0]:,}  rows × {test.shape[1]} columns")
train.head()

# %% [markdown]
# ## 2. Label distribution
#
# NSL-KDD has **dozens of specific attack names** (neptune, smurf, portsweep, etc.)
# We group them into 5 broad categories for classification.

# %%
train["attack_category"] = train["label"].str.lower().map(ATTACK_MAP).fillna("Other")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw label (too many to read — shows why we group)
top_labels = train["label"].value_counts().head(20)
axes[0].barh(top_labels.index[::-1], top_labels.values[::-1], color="#4F46E5")
axes[0].set_title("Top 20 raw attack labels", fontweight="bold")
axes[0].set_xlabel("Count")

# Grouped categories
cat_counts = train["attack_category"].value_counts()
colors = ["#10B981", "#EF4444", "#F59E0B", "#6366F1", "#EC4899"]
axes[1].bar(cat_counts.index, cat_counts.values, color=colors[:len(cat_counts)])
axes[1].set_title("Grouped attack categories", fontweight="bold")
axes[1].set_ylabel("Count")
for i, (cat, cnt) in enumerate(cat_counts.items()):
    axes[1].text(i, cnt + 500, f"{cnt:,}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("../plots/eda_label_distribution.png", bbox_inches="tight")
plt.show()

print("\nClass distribution:")
for cat, cnt in cat_counts.items():
    pct = cnt / len(train) * 100
    print(f"  {cat:8s}: {cnt:>7,}  ({pct:.1f}%)")

# %% [markdown]
# **Key insight**: The dataset is heavily imbalanced — Normal and DoS dominate.
# R2L and U2R are very rare, which makes them hard to detect (real-world challenge!).

# %% [markdown]
# ## 3. Feature distributions by attack type

# %%
numeric_features = ["duration", "src_bytes", "dst_bytes", "count",
                    "srv_count", "serror_rate", "rerror_rate",
                    "same_srv_rate", "dst_host_count", "dst_host_serror_rate"]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for ax, feat in zip(axes, numeric_features):
    for cat in train["attack_category"].unique():
        subset = train[train["attack_category"] == cat][feat]
        subset_clipped = np.clip(subset, 0, subset.quantile(0.99))
        ax.hist(subset_clipped, bins=30, alpha=0.5, label=cat, density=True)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.set_xlabel("")

axes[0].legend(fontsize=8, loc="upper right")
plt.suptitle("Feature distributions by attack category", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("../plots/eda_feature_distributions.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Correlation heatmap
# Find which numeric features are correlated — helps understand redundancy.

# %%
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "difficulty_level"]
corr = train[numeric_cols].corr()

# Show only highly correlated pairs (>0.7)
high_corr = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.7:
            high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

print("Highly correlated feature pairs (|r| > 0.7):")
for f1, f2, r in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
    print(f"  {f1:35s} ↔ {f2:35s}  r={r:.3f}")

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
            annot=False, ax=ax, linewidths=0.3, vmin=-1, vmax=1)
ax.set_title("Feature correlation matrix (lower triangle)", fontweight="bold")
plt.tight_layout()
plt.savefig("../plots/eda_correlation.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Protocol type and service analysis

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Protocol distribution per attack category
proto_cat = pd.crosstab(train["protocol_type"], train["attack_category"], normalize="index")
proto_cat.plot(kind="bar", ax=axes[0], colormap="Set2")
axes[0].set_title("Protocol type by category", fontweight="bold")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].legend(fontsize=8)

# Flag distribution
flag_counts = train["flag"].value_counts()
axes[1].bar(flag_counts.index, flag_counts.values, color="#6366F1")
axes[1].set_title("Connection flag distribution", fontweight="bold")
axes[1].tick_params(axis="x", rotation=30)

# Top services in attacks vs normal
attack_df = train[train["attack_category"] != "Normal"]
normal_df = train[train["attack_category"] == "Normal"]
top_attack_svc = attack_df["service"].value_counts().head(10)
top_normal_svc = normal_df["service"].value_counts().head(10)

x = np.arange(10)
axes[2].barh(top_attack_svc.index[::-1], top_attack_svc.values[::-1],
             alpha=0.7, label="Attack", color="#EF4444")
axes[2].set_title("Top services in attack traffic", fontweight="bold")

plt.tight_layout()
plt.savefig("../plots/eda_categorical.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Key statistics per attack category
# This table is great to include in your project README.

# %%
stats = train.groupby("attack_category").agg(
    count=("label", "count"),
    avg_duration=("duration", "mean"),
    avg_src_bytes=("src_bytes", "mean"),
    avg_dst_bytes=("dst_bytes", "mean"),
    avg_serror_rate=("serror_rate", "mean"),
    avg_rerror_rate=("rerror_rate", "mean"),
    avg_count=("count", "mean"),
).round(2)
print(stats.to_string())

# %% [markdown]
# ## 7. EDA summary — key observations
#
# Write your own observations here after running the cells above.
# These are the kinds of insights you should mention in interviews.
#
# **Template**:
# 1. DoS attacks have very high `count` and `serror_rate` values — SYN flood signature
# 2. Probe attacks use `icmp` and `udp` protocols more than normal traffic
# 3. R2L and U2R are rare but have distinct `logged_in` and `num_failed_logins` patterns
# 4. `serror_rate` and `dst_host_serror_rate` are highly correlated (>0.9) — redundant features
# 5. `SF` flag dominates Normal traffic; `S0` flag is almost exclusively DoS

print("\nSONA EDA complete! Check the plots/ folder for all visualizations.")
print("Now move on to: python src/train.py")
