"""
Simulate microbiome abundance data for transfer learning.

Source domain : Large unlabeled HMP-like healthy microbiome dataset
                (used to pretrain the autoencoder)
Target domain : Small labeled IBD vs Control dataset
                (used to fine-tune the classifier)

Biological basis: HMP profile; IBD taxa shifts from Franzosa et al. 2019.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

FIRMICUTES = [
    "Faecalibacterium_prausnitzii", "Roseburia_intestinalis", "Lachnospira_multipara",
    "Blautia_obeum", "Ruminococcus_gnavus", "Ruminococcus_bromii",
    "Eubacterium_rectale", "Eubacterium_hallii", "Anaerostipes_caccae",
    "Subdoligranulum_variabile", "Coprococcus_eutactus", "Dorea_longicatena",
]
BACTEROIDETES = [
    "Bacteroides_fragilis", "Bacteroides_thetaiotaomicron", "Bacteroides_vulgatus",
    "Bacteroides_uniformis", "Prevotella_copri", "Prevotella_stercorea",
    "Alistipes_shahii", "Alistipes_putredinis", "Parabacteroides_distasonis",
    "Phocaeicola_dorei",
]
ACTINOBACTERIA = [
    "Bifidobacterium_longum", "Bifidobacterium_adolescentis",
    "Collinsella_aerofaciens", "Eggerthella_lenta",
]
IBD_ENRICHED = [
    "Escherichia_coli", "Klebsiella_pneumoniae", "Haemophilus_parainfluenzae",
    "Fusobacterium_nucleatum", "Peptostreptococcus_stomatis",
    "Ruminococcus_torques",
]
IBD_DEPLETED = [
    "Akkermansia_muciniphila", "Bifidobacterium_bifidum",
    "Lactobacillus_rhamnosus", "Lactobacillus_acidophilus",
    "Christensenellaceae_sp", "Oscillospira_sp",
]

def _make_otu_names(n: int = 200) -> list[str]:
    named = FIRMICUTES + BACTEROIDETES + ACTINOBACTERIA + IBD_ENRICHED + IBD_DEPLETED
    extra = [f"OTU_{i:04d}" for i in range(n - len(named))]
    return (named + extra)[:n]

OTU_NAMES = _make_otu_names(200)
N_OTUS = len(OTU_NAMES)


def _clr_transform(X: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
    X_p = X + pseudocount
    log_X = np.log(X_p)
    return log_X - log_X.mean(axis=1, keepdims=True)


def simulate_source_domain(n_samples: int = 800, seed: int = 0) -> pd.DataFrame:
    """
    Large unlabeled healthy HMP-like microbiome dataset for autoencoder pretraining.
    Returns DataFrame (samples x OTUs), CLR-transformed.
    """
    rng = np.random.default_rng(seed)
    alpha = np.ones(N_OTUS) * 0.3
    for i, name in enumerate(OTU_NAMES):
        if name in FIRMICUTES:       alpha[i] = 2.5
        elif name in BACTEROIDETES:  alpha[i] = 2.0
        elif name in ACTINOBACTERIA: alpha[i] = 1.2
        elif name in IBD_DEPLETED:   alpha[i] = 1.5
        elif name in IBD_ENRICHED:   alpha[i] = 0.1

    raw = rng.dirichlet(alpha, size=n_samples)
    mask = rng.random(raw.shape) < 0.35
    raw[mask] = 0
    raw /= raw.sum(axis=1, keepdims=True) + 1e-10
    clr = _clr_transform(raw)
    return pd.DataFrame(clr, columns=OTU_NAMES,
                        index=[f"HMP_{i+1:04d}" for i in range(n_samples)])


def simulate_target_domain(
    n_samples: int = 120,
    n_ibd: Optional[int] = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Small labeled IBD vs Control dataset (target domain).
    Returns X (samples x OTUs CLR), y (0=Control, 1=IBD).
    """
    rng = np.random.default_rng(seed)
    if n_ibd is None:
        n_ibd = n_samples // 2
    n_ctrl = n_samples - n_ibd

    alpha_ctrl = np.ones(N_OTUS) * 0.3
    for i, name in enumerate(OTU_NAMES):
        if name in FIRMICUTES:       alpha_ctrl[i] = 2.5
        elif name in BACTEROIDETES:  alpha_ctrl[i] = 2.0
        elif name in ACTINOBACTERIA: alpha_ctrl[i] = 1.2
        elif name in IBD_DEPLETED:   alpha_ctrl[i] = 1.8
        elif name in IBD_ENRICHED:   alpha_ctrl[i] = 0.1

    alpha_ibd = alpha_ctrl.copy()
    for i, name in enumerate(OTU_NAMES):
        if name in IBD_ENRICHED:   alpha_ibd[i] = 3.0
        elif name in IBD_DEPLETED: alpha_ibd[i] = 0.2
        elif name in FIRMICUTES:   alpha_ibd[i] = 1.0

    ctrl_raw = rng.dirichlet(alpha_ctrl, size=n_ctrl)
    ibd_raw  = rng.dirichlet(alpha_ibd,  size=n_ibd)
    for raw in [ctrl_raw, ibd_raw]:
        mask = rng.random(raw.shape) < 0.4
        raw[mask] = 0
    ctrl_raw /= ctrl_raw.sum(axis=1, keepdims=True) + 1e-10
    ibd_raw  /= ibd_raw.sum(axis=1, keepdims=True) + 1e-10

    X_raw = np.vstack([ctrl_raw, ibd_raw])
    clr   = _clr_transform(X_raw)
    ids = [f"CTRL_{i+1:03d}" for i in range(n_ctrl)] + [f"IBD_{i+1:03d}" for i in range(n_ibd)]
    return (pd.DataFrame(clr, columns=OTU_NAMES, index=ids),
            pd.Series([0]*n_ctrl + [1]*n_ibd, index=ids, name="label"))
