#!/usr/bin/env python3
"""
Train a TB vs NTM classifier on Merlin embeddings and visualize separation.
Reads embeddings CSV produced by extract_embeddings.py.
Outputs metrics, ROC plot, t-SNE (and UMAP if installed).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore

    HAS_UMAP = True
except (ImportError, Exception):
    HAS_UMAP = False


def load_embeddings(csv_path: Path):
    df = pd.read_csv(csv_path)
    X = np.stack(df["embedding"].apply(json.loads).to_numpy())
    y = (df["group"] == "TB").astype(int).to_numpy()  # TB=1, NTM=0
    return df, X, y


def plot_roc(y_true, y_prob, out_png: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC TB vs NTM")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return auc


def plot_embed(X2d, y, title: str, out_png: Path):
    plt.figure()
    plt.scatter(X2d[:, 0], X2d[:, 1], c=y, cmap="coolwarm", s=8, alpha=0.7)
    plt.title(f"{title} (TB=1 red, NTM=0 blue)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    csv_path = Path(r"C:\Users\sbohar3\merlin\results\embeddings\ntm_tb_embeddings.csv")
    out_dir = csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df, X, y = load_embeddings(csv_path)
    print(f"Loaded {len(df)} embeddings: {sum(y)} TB, {len(y) - sum(y)} NTM")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["NTM", "TB"])

    # Save metrics
    metrics_txt = out_dir / "tb_vs_ntm_metrics.txt"
    with open(metrics_txt, "w") as f:
        f.write(f"ROC AUC: {auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Saved metrics to {metrics_txt}")

    # ROC plot
    plot_roc(y_test, y_prob, out_dir / "tb_vs_ntm_roc.png")

    # t-SNE (full dataset) - adjust perplexity for small n
    n_samples = len(X)
    perplexity = min(30, max(2, (n_samples - 1) // 3))
    if n_samples < 10:
        print(f"Warning: Only {n_samples} samples; t-SNE may not produce meaningful results.")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    X_tsne = tsne.fit_transform(X)
    plot_embed(X_tsne, y, "t-SNE", out_dir / "tb_vs_ntm_tsne.png")

    # UMAP (optional)
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X)
        plot_embed(X_umap, y, "UMAP", out_dir / "tb_vs_ntm_umap.png")

    print(f"Done. AUC={auc:.3f}. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
