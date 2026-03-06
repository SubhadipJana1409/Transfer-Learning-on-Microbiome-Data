# Day 21 · Transfer Learning on Microbiome Data

**Pretrained autoencoder embeddings → IBD classifier · Pure NumPy · 9 publication-quality figures**

Part of the [#30DaysOfBioinformatics](https://github.com/SubhadipJana1409) challenge.
Previous: [Day 20 – scRNA-seq Gut Clustering](https://github.com/SubhadipJana1409/day20-scrna-gut-clustering)

---

## Overview

A core challenge in microbiome ML is **data scarcity** — labeled clinical samples (IBD vs healthy) are expensive to collect, while large unlabeled microbiome datasets (HMP, MGnify) are abundant.

This project applies **transfer learning** to bridge that gap:

1. **Pretrain** a stacked autoencoder on 800 unlabeled HMP-like healthy microbiome samples (source domain)
2. **Extract** the encoder as a fixed feature extractor (32-dimensional latent space)
3. **Fine-tune** a classifier on just 90 labeled IBD samples (target domain)
4. **Compare** 5 strategies: transfer frozen, transfer fine-tuned, from scratch, Random Forest, Logistic Regression

**Key finding:** Transfer learning (AUC ~0.71) massively outperforms training from scratch (AUC ~0.39) on small labeled datasets — exactly the regime that matters clinically.

---

## Transfer Learning Architecture

```
SOURCE DOMAIN (800 unlabeled HMP samples)
          │
          ▼
┌─────────────────────────────┐
│  AUTOENCODER PRETRAINING    │
│  200-OTU CLR → 128 → 64    │
│       → latent(32)          │   ← Encoder weights saved
│       → 64 → 128 → 200     │
└─────────────────────────────┘
          │ encoder weights
          ▼
TARGET DOMAIN (90 labeled IBD samples)
          │
          ▼
┌──────────────────────────────┐
│  CLASSIFIER FINE-TUNING      │
│  Strategy A: frozen encoder  │  32 → 32 → sigmoid
│  Strategy B: fine-tune all   │  200 → 128 → 64 → 32 → sigmoid
│  Strategy C: from scratch    │  random init, same arch
└──────────────────────────────┘
```

---

## 5 Strategies Compared

| Strategy | Description |
|---|---|
| **Transfer (frozen)** | Pretrained encoder frozen; only classifier head trained |
| **Transfer (fine-tune)** | Pretrained encoder fine-tuned with lower LR |
| **From scratch** | Same architecture, randomly initialised |
| **Random Forest** | Ensemble on raw CLR OTU features |
| **Logistic Regression** | Regularised linear model on CLR features |

---

## Figures

| Figure | Description |
|--------|-------------|
| `fig1_domain_overview.png`   | PCA: source HMP vs target IBD domain overlap |
| `fig2_training_curves.png`   | Autoencoder pretraining loss (train + val) |
| `fig3_latent_space.png`      | 32-dim latent space projected to PCA — IBD vs Control separation |
| `fig4_reconstruction.png`    | Original vs reconstructed OTU abundances |
| `fig5_roc_curves.png`        | ROC curves for all 5 strategies |
| `fig6_confusion_matrices.png`| Confusion matrices for all strategies |
| `fig7_performance_bar.png`   | AUC / Accuracy / F1 grouped bar chart |
| `fig8_learning_curve.png`    | AUC vs training set size — transfer vs scratch |
| `fig9_summary.png`           | Ranking + latent dim variance + improvement over scratch |

---

## Quick Start

```bash
git clone https://github.com/SubhadipJana1409/Transfer-Learning-on-Microbiome-Data.git
cd day21-transfer-learning-microbiome
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
```

### Configuration (`configs/config.yaml`)

```yaml
data:
  n_source: 800       # unlabeled HMP-like samples for pretraining
  n_target: 120       # labeled IBD study samples

autoencoder:
  hidden_dims: [128, 64]
  latent_dim:  32
  epochs:      80

transfer:
  epochs:      80
  lr:          0.0005
```

---

## Project Structure

```
day21-transfer-learning-microbiome/
├── src/
│   ├── data/
│   │   └── simulator.py          # HMP source + IBD target domain simulator
│   ├── models/
│   │   ├── autoencoder.py        # Stacked autoencoder (pure NumPy)
│   │   └── classifier.py         # Transfer learning + baseline classifiers
│   ├── visualization/
│   │   └── plots.py              # All 9 figures
│   └── main.py
├── tests/
│   ├── test_simulator.py         # 9 tests
│   ├── test_autoencoder.py       # 11 tests
│   └── test_classifier_plots.py  # 13 tests
├── configs/config.yaml
├── outputs/                      # Figures, metrics, saved model
└── requirements.txt
```

---

## Tests

```bash
pytest tests/ -v
# 33 passed
```

---

## Methods

**Data**: Microbiome composition simulated as Dirichlet-distributed relative abundances with zero-inflation, CLR-transformed to handle compositionality. Source domain mimics the HMP healthy gut profile; target domain introduces IBD-associated taxa shifts (Franzosa et al. 2019).

**Autoencoder**: Stacked encoder 200→128→64→32 with ReLU activations, He initialisation, MSE loss, mini-batch SGD with gradient clipping (norm ≤ 1.0), and input z-score normalisation. Trained on the source domain only (unsupervised).

**Transfer classifier**: A two-layer MLP head (32→32→sigmoid) trained on the encoded latent features with binary cross-entropy loss.

**Evaluation**: Stratified 75/25 train-test split on the target domain. Metrics: AUC-ROC, Accuracy, F1.

---

## References

1. Franzosa EA et al. (2019). Gut microbiome structure and metabolic activity in inflammatory bowel disease. *Nature Microbiology*.
2. The Human Microbiome Project Consortium (2012). Structure, function and diversity of the healthy human microbiome. *Nature*, 486, 207–214.
3. Aitchison J (1982). The statistical analysis of compositional data. *JRSS-B*.

---

## License

MIT
