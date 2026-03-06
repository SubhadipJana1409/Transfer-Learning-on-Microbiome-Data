"""
Day 21 · Transfer Learning on Microbiome Data
=============================================
Source domain: Large unlabeled HMP-like healthy microbiome (800 samples)
Target domain: Small labeled IBD vs Control dataset (120 samples)

Pipeline
--------
1. Simulate source (HMP) + target (IBD) microbiome data
2. Pretrain autoencoder on source domain (unsupervised)
3. Fine-tune classifier on target domain (5 strategies)
4. Evaluate + learning curve analysis
5. Generate 9 publication-quality figures

Usage
-----
    python -m src.main
    python -m src.main --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.simulator     import simulate_source_domain, simulate_target_domain, OTU_NAMES
from src.models.autoencoder import MicrobiomeAutoencoder
from src.models.classifier  import TransferClassifier
from src.visualization.plots import generate_all
from src.utils.logger       import setup_logging
from src.utils.config       import load_config

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Transfer learning on microbiome data")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--quiet",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    setup_logging(level=logging.WARNING if args.quiet else logging.INFO)
    out  = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Day 21 · Transfer Learning on Microbiome Data")
    logger.info("=" * 60)

    data_cfg = cfg.get("data", {})
    ae_cfg   = cfg.get("autoencoder", {})
    tl_cfg   = cfg.get("transfer", {})

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    logger.info("[1/6] Simulating microbiome data …")
    X_source = simulate_source_domain(
        n_samples=data_cfg.get("n_source", 800),
        seed=data_cfg.get("seed", 0),
    ).values

    X_target, y_target_s = simulate_target_domain(
        n_samples=data_cfg.get("n_target", 120),
        seed=data_cfg.get("seed", 42) + 1,
    )
    X_target = X_target.values
    y_target = y_target_s.values

    # Train/test split on target domain
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_target, y_target,
        test_size=0.25, stratify=y_target,
        random_state=data_cfg.get("seed", 42),
    )
    logger.info("Source: %d samples  |  Target train: %d  test: %d",
                len(X_source), len(X_tr), len(X_te))

    # ── Step 2: Pretrain autoencoder ──────────────────────────────────────────
    logger.info("[2/6] Pretraining autoencoder on source domain …")
    n_val = int(len(X_source) * 0.1)
    X_src_val  = X_source[:n_val]
    X_src_train = X_source[n_val:]

    ae = MicrobiomeAutoencoder(
        input_dim=X_source.shape[1],
        hidden_dims=tuple(ae_cfg.get("hidden_dims", [128, 64])),
        latent_dim=ae_cfg.get("latent_dim", 32),
        lr=ae_cfg.get("lr", 1e-3),
        seed=0,
    )
    ae.fit(
        X_src_train, X_val=X_src_val,
        epochs=ae_cfg.get("epochs", 80),
        batch_size=ae_cfg.get("batch_size", 64),
    )
    ae.save(out / "models" / "autoencoder.joblib")

    # ── Step 3: Train all classifiers ────────────────────────────────────────
    logger.info("[3/6] Training classifiers …")
    tc = TransferClassifier(ae, seed=42)
    tc.fit_transfer_frozen(X_tr, y_tr,
                           epochs=tl_cfg.get("epochs", 80),
                           lr=tl_cfg.get("lr", 5e-4))
    tc.fit_transfer_finetune(X_tr, y_tr,
                             epochs=tl_cfg.get("epochs", 80))
    tc.fit_scratch(X_tr, y_tr,
                   epochs=tl_cfg.get("epochs", 80))
    tc.fit_baselines(X_tr, y_tr)

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    logger.info("[4/6] Evaluating …")
    results = tc.evaluate(X_te, y_te)

    # ── Step 5: Learning curve ────────────────────────────────────────────────
    logger.info("[5/6] Computing learning curve …")
    lc = tc.learning_curve(
        X_tr, y_tr, X_te, y_te,
        train_sizes=tl_cfg.get("train_sizes", [8, 15, 25, 40, 60, 80]),
    )

    # ── Step 6: Figures + outputs ─────────────────────────────────────────────
    logger.info("[6/6] Generating figures …")
    generate_all(X_source, X_target, y_target, ae, results, lc, out, OTU_NAMES)

    # Save metrics table
    rows = []
    for strat, res in results.items():
        rows.append({
            "strategy": strat,
            "accuracy": res["accuracy"],
            "auc":      res["auc"],
            "f1":       res["f1"],
        })
    pd.DataFrame(rows).to_csv(out / "model_metrics.csv", index=False)

    pd.DataFrame({
        "train_size":    lc["train_sizes"],
        "auc_transfer":  lc["auc_transfer"],
        "auc_scratch":   lc["auc_scratch"],
    }).to_csv(out / "learning_curve.csv", index=False)

    elapsed = time.time() - t0
    best = max(results, key=lambda s: results[s]["auc"])

    print("\n" + "="*52)
    print("  Day 21 · Transfer Learning Summary")
    print("="*52)
    print(f"  Source domain  : {len(X_source):,} samples (unlabeled HMP)")
    print(f"  Target domain  : {len(X_target)} samples (IBD study)")
    print(f"  AE latent dim  : {ae.latent_dim}")
    print(f"  AE train loss  : {ae.train_losses[-1]:.4f}")
    print()
    for strat, res in sorted(results.items(), key=lambda x: -x[1]["auc"]):
        flag = " ← best" if strat == best else ""
        print(f"  {LABEL_MAP.get(strat, strat):<28}  AUC={res['auc']:.3f}  Acc={res['accuracy']:.3f}{flag}")
    print(f"\n  Figures        : 9 saved to {out}/")
    print(f"  Elapsed        : {elapsed:.1f}s")
    print("="*52 + "\n")


LABEL_MAP = {
    "transfer_frozen":   "Transfer (frozen)",
    "transfer_finetune": "Transfer (fine-tune)",
    "scratch":           "From scratch",
    "baseline_rf":       "Random Forest",
    "baseline_lr":       "Logistic Regression",
}

if __name__ == "__main__":
    main()
