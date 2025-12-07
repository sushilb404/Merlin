#!/usr/bin/env python3
"""
Comprehensive embedding analysis and visualization.
Generates 12+ visualizations from TB vs NTM embeddings.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import umap

    HAS_UMAP = True
except (ImportError, Exception):
    HAS_UMAP = False

# Style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

OUTPUT_DIR = Path(r"C:\Users\sbohar3\merlin\results\embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_embeddings(csv_path: Path):
    """Load embeddings from CSV."""
    df = pd.read_csv(csv_path, encoding='utf-16')
    embeddings = []
    for _, row in df.iterrows():
        emb = json.loads(row["embedding"])
        embeddings.append(emb)
    X = np.array(embeddings)
    y = (df["group"] == "TB").astype(int).values
    groups = df["group"].values
    return X, y, groups


def plot_confusion_matrix(y_test, y_pred):
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.xticks([0.5, 1.5], ["NTM", "TB"])
    plt.yticks([0.5, 1.5], ["NTM", "TB"])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved confusion_matrix.png")


def plot_precision_recall(y_test, y_scores):
    """Precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label="Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved precision_recall_curve.png")


def plot_3d_tsne(X, y, groups):
    """3D t-SNE projection."""
    print("Computing 3D t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, max(2, (len(X) - 1) // 3)))
    X_3d = tsne.fit_transform(X)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    for label, color, name in [(0, "blue", "NTM"), (1, "red", "TB")]:
        mask = y == label
        ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], c=color, label=name, s=20, alpha=0.6)
    
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    ax.set_title("3D t-SNE Projection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tsne_3d.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved tsne_3d.png")


def plot_3d_umap(X, y, groups):
    """3D UMAP projection."""
    if not HAS_UMAP:
        print("⚠️  UMAP not available, skipping 3D UMAP")
        return
    
    print("Computing 3D UMAP (this may take a minute)...")
    reducer = umap.UMAP(n_components=3, random_state=42)
    X_3d = reducer.fit_transform(X)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    for label, color, name in [(0, "blue", "NTM"), (1, "red", "TB")]:
        mask = y == label
        ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], c=color, label=name, s=20, alpha=0.6)
    
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.set_title("3D UMAP Projection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "umap_3d.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved umap_3d.png")


def plot_pca(X, y):
    """PCA projection with variance explained."""
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Variance explained
    ax1.plot(cumsum, linewidth=2)
    ax1.axhline(y=0.95, color="r", linestyle="--", label="95% variance")
    ax1.set_xlabel("Number of Components")
    ax1.set_ylabel("Cumulative Variance Explained")
    ax1.set_title("PCA Variance Explained")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2D projection
    for label, color, name in [(0, "blue", "NTM"), (1, "red", "TB")]:
        mask = y == label
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name, s=20, alpha=0.6)
    
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax2.set_title("PCA 2D Projection")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved pca_analysis.png")


def plot_kmeans_clustering(X, y):
    """K-means clustering analysis."""
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centers_pca = pca.transform(kmeans.cluster_centers_)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Predicted clusters
    for cluster in range(2):
        mask = cluster_labels == cluster
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {cluster}", s=20, alpha=0.6)
    ax1.scatter(centers_pca[:, 0], centers_pca[:, 1], c="black", marker="X", s=200, label="Centroids")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax1.set_title("K-Means Clustering")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Ground truth
    for label, color, name in [(0, "blue", "NTM"), (1, "red", "TB")]:
        mask = y == label
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name, s=20, alpha=0.6)
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax2.set_title("Ground Truth (TB vs NTM)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kmeans_clustering.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved kmeans_clustering.png")


def plot_embedding_distributions(X, y):
    """Distribution of top embedding dimensions."""
    # Get top 4 dimensions by variance
    top_dims = np.argsort(np.var(X, axis=0))[-4:]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, dim in enumerate(top_dims):
        ax = axes[idx]
        for label, color, name in [(0, "blue", "NTM"), (1, "red", "TB")]:
            mask = y == label
            ax.hist(X[mask, dim], bins=30, alpha=0.6, label=name, color=color)
        ax.set_xlabel(f"Dimension {dim}")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Embedding Distribution (Dim {dim})")
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "embedding_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved embedding_distributions.png")


def plot_confidence_distribution(y_test, y_scores):
    """Confidence score distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ntm_scores = y_scores[y_test == 0]
    tb_scores = y_scores[y_test == 1]
    
    ax.hist(ntm_scores, bins=30, alpha=0.6, label="NTM", color="blue")
    ax.hist(tb_scores, bins=30, alpha=0.6, label="TB", color="red")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Model Confidence Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confidence_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved confidence_distribution.png")


def plot_feature_importance(X, y, X_scaler):
    """Top discriminative embedding dimensions."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaler.fit_transform(X), y)
    
    coef = np.abs(clf.coef_[0])
    top_indices = np.argsort(coef)[-10:]
    top_coefs = coef[top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_indices)), top_coefs, color="steelblue")
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([f"Dim {i}" for i in top_indices])
    ax.set_xlabel("Absolute Coefficient")
    ax.set_title("Top 10 Discriminative Embedding Dimensions")
    ax.grid(alpha=0.3, axis="x")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved feature_importance.png")


def plot_embedding_space_distances(X, y):
    """Embedding space separation metrics."""
    from scipy.spatial.distance import cdist
    
    # Compute centroid distances
    ntm_center = X[y == 0].mean(axis=0)
    tb_center = X[y == 1].mean(axis=0)
    centroid_dist = np.linalg.norm(ntm_center - tb_center)
    
    # Intra-class distances
    ntm_distances = cdist(X[y == 0], [ntm_center])[0]
    tb_distances = cdist(X[y == 1], [tb_center])[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distance distributions
    ax = axes[0]
    ax.hist(ntm_distances, bins=30, alpha=0.6, label="NTM", color="blue")
    ax.hist(tb_distances, bins=30, alpha=0.6, label="TB", color="red")
    ax.set_xlabel("Distance to Centroid")
    ax.set_ylabel("Frequency")
    ax.set_title("Intra-class Distance Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Separation summary
    ax = axes[1]
    metrics = {
        "Centroid\nDistance": centroid_dist,
        "NTM Avg\nIntra-dist": ntm_distances.mean(),
        "TB Avg\nIntra-dist": tb_distances.mean(),
    }
    colors_bar = ["green", "blue", "red"]
    ax.bar(metrics.keys(), metrics.values(), color=colors_bar, alpha=0.7)
    ax.set_ylabel("Distance")
    ax.set_title("Embedding Space Separation Metrics")
    ax.grid(alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "embedding_distances.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved embedding_distances.png")


def plot_misclassified_analysis(y_test, y_pred, X_test, y_scores):
    """Analyze misclassified cases."""
    misclassified_idx = np.where(y_test != y_pred)[0]
    
    if len(misclassified_idx) == 0:
        print("⚠️  No misclassified cases found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Misclassified confidence
    ax = axes[0]
    mc_scores = y_scores[misclassified_idx]
    mc_true = y_test[misclassified_idx]
    
    for label, color, name in [(0, "blue", "NTM"), (1, "red", "TB")]:
        mask = mc_true == label
        ax.scatter(np.arange(len(mc_true))[mask], mc_scores[mask], c=color, label=f"{name} (misclassified)", s=50)
    ax.axhline(y=0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_ylabel("Predicted Probability")
    ax.set_title(f"Misclassified Cases ({len(misclassified_idx)} total)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Error types
    ax = axes[1]
    false_positives = np.sum((y_test == 0) & (y_pred == 1))
    false_negatives = np.sum((y_test == 1) & (y_pred == 0))
    true_positives = np.sum((y_test == 1) & (y_pred == 1))
    true_negatives = np.sum((y_test == 0) & (y_pred == 0))
    
    categories = ["True\nNegatives", "True\nPositives", "False\nPositives", "False\nNegatives"]
    values = [true_negatives, true_positives, false_positives, false_negatives]
    colors_err = ["green", "green", "red", "red"]
    
    ax.bar(categories, values, color=colors_err, alpha=0.7)
    ax.set_ylabel("Count")
    ax.set_title("Classification Results Summary")
    ax.grid(alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "misclassified_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved misclassified_analysis.png")


def plot_correlation_heatmap(X, y):
    """Correlation of top embedding dimensions."""
    # Top 10 dimensions
    top_dims = np.argsort(np.var(X, axis=0))[-10:]
    X_top = X[:, top_dims]
    
    corr = np.corrcoef(X_top.T)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, cbar=True, ax=ax)
    ax.set_xticklabels([f"D{i}" for i in top_dims])
    ax.set_yticklabels([f"D{i}" for i in top_dims])
    ax.set_title("Correlation Matrix (Top 10 Dimensions)")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved correlation_heatmap.png")


def main():
    # Try multiple locations
    possible_paths = [
        Path(r"C:\Users\sbohar3\Downloads\BMI_555_Sushil\results\embeddings\ntm_tb_embeddings.csv"),
        Path(r"C:\Users\sbohar3\merlin\results\embeddings\ntm_tb_embeddings.csv"),
    ]
    
    csv_path = None
    for p in possible_paths:
        if p.exists():
            csv_path = p
            break
    
    if csv_path is None:
        print(f"Error: embeddings CSV not found in any location")
        return
    
    print("Loading embeddings...")
    X, y, groups = load_embeddings(csv_path)
    print(f"Loaded {len(X)} embeddings")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    print("Training classifier...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    y_scores = clf.predict_proba(X_test_scaled)[:, 1]
    
    auc = roc_auc_score(y_test, y_scores)
    print(f"AUC: {auc:.3f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_test, y_pred)
    plot_precision_recall(y_test, y_scores)
    plot_3d_tsne(X_train, y_train, groups[:len(y_train)])
    plot_3d_umap(X_train, y_train, groups[:len(y_train)])
    plot_pca(X_train, y_train)
    plot_kmeans_clustering(X_train, y_train)
    plot_embedding_distributions(X_train, y_train)
    plot_confidence_distribution(y_test, y_scores)
    plot_feature_importance(X_train, y_train, StandardScaler())
    plot_embedding_space_distances(X, y)
    plot_misclassified_analysis(y_test, y_pred, X_test, y_scores)
    plot_correlation_heatmap(X_train, y_train)
    
    print(f"\n✨ All visualizations saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
