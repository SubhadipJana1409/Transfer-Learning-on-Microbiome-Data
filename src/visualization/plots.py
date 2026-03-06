"""
9 publication-quality figures for transfer learning on microbiome data.

fig1  : Source vs target domain PCA overview
fig2  : Autoencoder training & validation loss curves
fig3  : Latent space UMAP/PCA — coloured by condition
fig4  : Reconstruction quality (input vs reconstructed for top OTUs)
fig5  : ROC curves — all 5 strategies
fig6  : Confusion matrices (2×2 grid)
fig7  : Performance bar chart (AUC, Accuracy, F1)
fig8  : Learning curve — AUC vs training set size (transfer vs scratch)
fig9  : Summary — strategy comparison + latent dim importance
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

PALETTE = {
    "transfer_frozen":   "#2ECC71",
    "transfer_finetune": "#27AE60",
    "scratch":           "#E74C3C",
    "baseline_rf":       "#3498DB",
    "baseline_lr":       "#9B59B6",
}
LABEL_MAP = {
    "transfer_frozen":   "Transfer (frozen)",
    "transfer_finetune": "Transfer (fine-tune)",
    "scratch":           "From scratch",
    "baseline_rf":       "Random Forest",
    "baseline_lr":       "Logistic Regression",
}
COND_PAL = {"Control": "#3498DB", "IBD": "#E74C3C"}
DPI = 150


def _save(fig, out_dir: Path, name: str) -> None:
    p = out_dir / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)


# ── Fig 1: Domain overview (PCA) ──────────────────────────────────────────────
def fig1_domain_overview(
    X_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    out_dir: Path,
) -> None:
    pca = PCA(n_components=2)
    X_all = np.vstack([X_source[:200], X_target])
    coords = pca.fit_transform(X_all)
    n_src = min(200, len(X_source))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Domain Overview: Source (HMP) vs Target (IBD Study)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.scatter(coords[:n_src, 0], coords[:n_src, 1],
               c="#95A5A6", s=12, alpha=0.4, label=f"Source HMP (n={n_src})")
    ax.scatter(coords[n_src:, 0], coords[n_src:, 1],
               c=["#3498DB" if l==0 else "#E74C3C" for l in y_target],
               s=40, alpha=0.85, edgecolors="white", lw=0.5,
               zorder=3, label="Target (IBD study)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Source vs Target Domain Overlap")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top","right"]].set_visible(False)

    ax = axes[1]
    target_coords = coords[n_src:]
    for label, cond, col in [(0,"Control","#3498DB"),(1,"IBD","#E74C3C")]:
        idx = y_target == label
        ax.scatter(target_coords[idx,0], target_coords[idx,1],
                   c=col, s=60, alpha=0.8, label=cond,
                   edgecolors="white", lw=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Target Domain: IBD vs Control")
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, out_dir, "fig1_domain_overview.png")


# ── Fig 2: Autoencoder training curves ────────────────────────────────────────
def fig2_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(train_losses)+1)
    ax.plot(epochs, train_losses, color="#2C3E50", lw=2, label="Train loss")
    if val_losses:
        ax.plot(epochs, val_losses, color="#E74C3C", lw=2, ls="--", label="Val loss")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.set_title("Autoencoder Pretraining Loss (Source Domain — HMP)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig2_training_curves.png")


# ── Fig 3: Latent space ────────────────────────────────────────────────────────
def fig3_latent_space(
    Z_target: np.ndarray,
    y_target: np.ndarray,
    Z_source: np.ndarray,
    out_dir: Path,
) -> None:
    pca = PCA(n_components=2)
    Z_all = np.vstack([Z_source[:300], Z_target])
    coords = pca.fit_transform(Z_all)
    n_src = min(300, len(Z_source))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Pretrained Autoencoder Latent Space (32 dimensions → PCA)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.scatter(coords[:n_src,0], coords[:n_src,1],
               c="#BDC3C7", s=10, alpha=0.3, label=f"HMP source (n={n_src})")
    ax.scatter(coords[n_src:,0], coords[n_src:,1],
               c=["#3498DB" if l==0 else "#E74C3C" for l in y_target],
               s=50, alpha=0.85, edgecolors="white", lw=0.5, zorder=3,
               label="IBD target")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("All samples in latent space")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top","right"]].set_visible(False)

    ax = axes[1]
    tc = coords[n_src:]
    for label, cond, col in [(0,"Control","#3498DB"),(1,"IBD","#E74C3C")]:
        idx = y_target == label
        ax.scatter(tc[idx,0], tc[idx,1], c=col, s=70, alpha=0.85,
                   label=cond, edgecolors="white", lw=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Target domain separation in latent space")
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig3_latent_space.png")


# ── Fig 4: Reconstruction quality ─────────────────────────────────────────────
def fig4_reconstruction(
    X_orig: np.ndarray,
    X_recon: np.ndarray,
    otu_names: list[str],
    out_dir: Path,
    n_show: int = 12,
) -> None:
    mean_orig  = X_orig.mean(axis=0)
    mean_recon = X_recon.mean(axis=0)
    var_orig   = X_orig.var(axis=0)
    top_idx    = np.argsort(var_orig)[-n_show:]
    labels     = [otu_names[i].replace("_", " ") for i in top_idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Autoencoder Reconstruction Quality (Top Variable OTUs)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    x = np.arange(n_show)
    w = 0.35
    ax.bar(x - w/2, mean_orig[top_idx],  w, label="Original",     color="#2C3E50", alpha=0.8)
    ax.bar(x + w/2, mean_recon[top_idx], w, label="Reconstructed", color="#E74C3C", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean CLR abundance")
    ax.set_title("Mean Abundance: Original vs Reconstructed")
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top","right"]].set_visible(False)

    ax = axes[1]
    ax.scatter(mean_orig, mean_recon, s=15, alpha=0.5, color="#3498DB")
    lim = [min(mean_orig.min(), mean_recon.min()) - 0.1,
           max(mean_orig.max(), mean_recon.max()) + 0.1]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.7, label="y = x (perfect)")
    corr = np.corrcoef(mean_orig, mean_recon)[0,1]
    ax.set_xlabel("Original mean CLR")
    ax.set_ylabel("Reconstructed mean CLR")
    ax.set_title(f"Mean Abundance Correlation (r={corr:.3f})")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, out_dir, "fig4_reconstruction.png")


# ── Fig 5: ROC curves ─────────────────────────────────────────────────────────
def fig5_roc_curves(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for strat, res in results.items():
        col   = PALETTE.get(strat, "gray")
        label = f"{LABEL_MAP.get(strat, strat)} (AUC={res['auc']:.3f})"
        ax.plot(res["fpr"], res["tpr"], color=col, lw=2.0, label=label)
    ax.plot([0,1], [0,1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves: Transfer Learning vs Baselines", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig5_roc_curves.png")


# ── Fig 6: Confusion matrices ─────────────────────────────────────────────────
def fig6_confusion_matrices(results: dict, out_dir: Path) -> None:
    strategies = list(results.keys())
    n = len(strategies)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
    axes = np.array(axes).ravel()
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

    for i, strat in enumerate(strategies):
        cm  = results[strat]["confusion"]
        acc = results[strat]["accuracy"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Control","IBD"],
                    yticklabels=["Control","IBD"],
                    ax=axes[i], cbar=False, linewidths=0.5)
        axes[i].set_title(f"{LABEL_MAP.get(strat, strat)}\nAcc={acc:.3f}",
                          fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Predicted", fontsize=9)
        axes[i].set_ylabel("True", fontsize=9)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig6_confusion_matrices.png")


# ── Fig 7: Performance bar chart ──────────────────────────────────────────────
def fig7_performance_bar(results: dict, out_dir: Path) -> None:
    metrics  = ["auc", "accuracy", "f1"]
    labels   = ["AUC-ROC", "Accuracy", "F1 Score"]
    strats   = list(results.keys())
    x        = np.arange(len(strats))
    width    = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for j, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [results[s][metric] for s in strats]
        bars = ax.bar(x + j*width, vals, width,
                      label=label, alpha=0.85, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                    f"{v:.3f}", ha="center", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels([LABEL_MAP.get(s, s) for s in strats], rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig7_performance_bar.png")


# ── Fig 8: Learning curve ─────────────────────────────────────────────────────
def fig8_learning_curve(lc: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sizes = lc["train_sizes"]
    ax.plot(sizes, lc["auc_transfer"], "o-",
            color=PALETTE["transfer_frozen"], lw=2.5, ms=7,
            label="Transfer (frozen encoder)")
    ax.plot(sizes, lc["auc_scratch"], "s--",
            color=PALETTE["scratch"], lw=2.5, ms=7,
            label="From scratch")
    ax.axhline(0.5, color="gray", lw=1, ls=":", alpha=0.6, label="Random baseline")
    ax.set_xlabel("Training Set Size (labeled samples)", fontsize=11)
    ax.set_ylabel("AUC-ROC on Test Set", fontsize=11)
    ax.set_title("Learning Curve: Transfer Learning Advantage\n(especially with small labeled data)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, frameon=False)
    ax.set_ylim(0.4, 1.05)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig8_learning_curve.png")


# ── Fig 9: Summary ─────────────────────────────────────────────────────────────
def fig9_summary(
    results: dict,
    autoencoder,
    X_source: np.ndarray,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Transfer Learning Summary: Microbiome IBD Classification",
                 fontsize=13, fontweight="bold")

    # Panel A: AUC bar (ranked)
    ax = axes[0]
    sorted_strats = sorted(results.keys(), key=lambda s: results[s]["auc"], reverse=True)
    aucs   = [results[s]["auc"]  for s in sorted_strats]
    colors = [PALETTE.get(s, "gray") for s in sorted_strats]
    bars = ax.barh([LABEL_MAP.get(s,s) for s in sorted_strats], aucs,
                   color=colors, edgecolor="white")
    for b, v in zip(bars, aucs):
        ax.text(v+0.002, b.get_y()+b.get_height()/2,
                f"{v:.3f}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("AUC-ROC", fontsize=10)
    ax.set_title("Model Ranking by AUC", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)

    # Panel B: Latent dim variance explained (encoder output variance per dim)
    ax = axes[1]
    Z_src = autoencoder.encode(X_source[:300])
    var_per_dim = Z_src.var(axis=0)
    sorted_idx  = np.argsort(var_per_dim)[::-1][:16]
    ax.bar(range(len(sorted_idx)), var_per_dim[sorted_idx],
           color="#2C3E50", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Latent Dimension (sorted by variance)", fontsize=10)
    ax.set_ylabel("Variance", fontsize=10)
    ax.set_title("Latent Dimension Variance\n(encoder feature importance proxy)", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)

    # Panel C: Improvement of transfer over scratch
    ax = axes[2]
    scratch_auc = results.get("scratch", {}).get("auc", np.nan)
    improvements = {
        LABEL_MAP.get(s, s): round(results[s]["auc"] - scratch_auc, 4)
        for s in sorted_strats if s != "scratch"
    }
    imp_labels = list(improvements.keys())
    imp_vals   = list(improvements.values())
    bar_colors = ["#2ECC71" if v >= 0 else "#E74C3C" for v in imp_vals]
    ax.barh(imp_labels, imp_vals, color=bar_colors, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("ΔAUC vs From Scratch", fontsize=10)
    ax.set_title("Improvement over\nFrom-Scratch Baseline", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, out_dir, "fig9_summary.png")


# ── Driver ─────────────────────────────────────────────────────────────────────
def generate_all(
    X_source, X_target, y_target,
    autoencoder, results, lc,
    out_dir: Path,
    otu_names: list[str],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating figures → %s", out_dir)

    Z_target = autoencoder.encode(X_target)
    Z_source = autoencoder.encode(X_source[:400])
    X_recon  = autoencoder.decode(Z_source)

    fig1_domain_overview(X_source, X_target, y_target, out_dir)
    fig2_training_curves(autoencoder.train_losses, autoencoder.val_losses, out_dir)
    fig3_latent_space(Z_target, y_target, Z_source, out_dir)
    fig4_reconstruction(X_source[:400], X_recon, otu_names, out_dir)
    fig5_roc_curves(results, out_dir)
    fig6_confusion_matrices(results, out_dir)
    fig7_performance_bar(results, out_dir)
    fig8_learning_curve(lc, out_dir)
    fig9_summary(results, autoencoder, X_source, out_dir)
    logger.info("All 9 figures saved.")
