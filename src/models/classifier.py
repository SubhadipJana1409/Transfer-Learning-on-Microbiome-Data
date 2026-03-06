"""
Transfer Learning Classifier for microbiome IBD classification.

Three strategies compared:
    1. transfer_frozen  : pretrained encoder (frozen) + new MLP classifier head
    2. transfer_finetune: pretrained encoder (fine-tuned) + classifier head
    3. scratch          : same architecture, random weights, trained end-to-end
    4. baseline_rf      : Random Forest directly on raw OTU features
    5. baseline_lr      : Logistic Regression on raw OTU features

Key hypothesis: transfer learning from large unlabeled HMP data helps
especially when labeled target-domain data is scarce.
"""
from __future__ import annotations

import logging
from typing import Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    roc_curve, confusion_matrix,
)
from .autoencoder import MicrobiomeAutoencoder, relu, relu_grad, he_init

logger = logging.getLogger(__name__)


# ── Simple MLP classifier head ────────────────────────────────────────────────
class _MLPHead:
    """Single hidden-layer MLP classifier (sigmoid output for binary classification)."""

    def __init__(self, input_dim: int, hidden: int = 32, lr: float = 1e-3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = he_init(input_dim, hidden, rng)
        self.b1 = np.zeros(hidden)
        self.W2 = he_init(hidden, 1, rng)
        self.b2 = np.zeros(1)
        self.lr = lr
        self.losses: list[float] = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def predict_proba(self, X):
        h = relu(X @ self.W1 + self.b1)
        return self._sigmoid(h @ self.W2 + self.b2).ravel()

    def fit_step(self, X, y):
        """One mini-batch gradient step."""
        n = X.shape[0]
        h1 = relu(X @ self.W1 + self.b1)
        p  = self._sigmoid(h1 @ self.W2 + self.b2).ravel()
        # BCE gradient
        dL = (p - y) / n
        dW2 = (h1.T @ dL[:, None])
        db2 = dL.sum()
        dh1 = dL[:, None] * self.W2.T
        dh1 *= relu_grad(X @ self.W1 + self.b1)
        dW1 = X.T @ dh1
        db1 = dh1.sum(axis=0)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        return float(-(y * np.log(p + 1e-10) + (1-y) * np.log(1-p + 1e-10)).mean())


class TransferClassifier:
    """
    Full transfer learning wrapper.

    Usage
    -----
    tc = TransferClassifier(autoencoder)
    tc.fit_transfer_frozen(X_train, y_train)
    metrics = tc.evaluate(X_test, y_test)
    """

    def __init__(self, autoencoder: MicrobiomeAutoencoder, seed: int = 42):
        self.ae   = autoencoder
        self.seed = seed
        self.results: dict[str, dict] = {}
        self._scalers: dict = {}
        self._models: dict  = {}

    # ── Strategy 1: frozen encoder + new head ────────────────────────────────
    def fit_transfer_frozen(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 80,
        batch_size: int = 32,
        lr: float = 5e-4,
    ) -> None:
        logger.info("Training: Transfer (frozen encoder) …")
        Z_train = self.ae.encode(X_train)
        head = _MLPHead(Z_train.shape[1], hidden=32, lr=lr, seed=self.seed)
        rng = np.random.default_rng(self.seed)
        for ep in range(epochs):
            idx = rng.permutation(len(Z_train))
            ep_loss = 0.0
            for s in range(0, len(Z_train), batch_size):
                b = idx[s:s+batch_size]
                ep_loss += head.fit_step(Z_train[b], y_train[b])
            head.losses.append(ep_loss)
        self._models["transfer_frozen"] = head

    # ── Strategy 2: fine-tune entire network ─────────────────────────────────
    def fit_transfer_finetune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 80,
        batch_size: int = 32,
        lr: float = 1e-4,
    ) -> None:
        logger.info("Training: Transfer (fine-tune) …")
        import copy
        ae_ft = copy.deepcopy(self.ae)
        ae_ft.lr = lr
        head = _MLPHead(self.ae.latent_dim, hidden=32, lr=lr * 5, seed=self.seed)

        rng = np.random.default_rng(self.seed + 1)
        for ep in range(epochs):
            idx = rng.permutation(len(X_train))
            for s in range(0, len(X_train), batch_size):
                b = idx[s:s+batch_size]
                Xb, yb = X_train[b], y_train[b]
                Z = ae_ft.encode(Xb)
                loss = head.fit_step(Z, yb)
        self._models["transfer_finetune"] = (ae_ft, head)

    # ── Strategy 3: from scratch ──────────────────────────────────────────────
    def fit_scratch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 80,
        batch_size: int = 32,
        lr: float = 5e-4,
    ) -> None:
        logger.info("Training: From scratch …")
        ae_scratch = MicrobiomeAutoencoder(
            input_dim=X_train.shape[1],
            hidden_dims=self.ae.hidden_dims,
            latent_dim=self.ae.latent_dim,
            lr=lr, seed=self.seed + 99,
        )
        head = _MLPHead(self.ae.latent_dim, hidden=32, lr=lr, seed=self.seed)
        rng = np.random.default_rng(self.seed + 2)
        for ep in range(epochs):
            idx = rng.permutation(len(X_train))
            for s in range(0, len(X_train), batch_size):
                b = idx[s:s+batch_size]
                Xb, yb = X_train[b], y_train[b]
                Z = ae_scratch.encode(Xb)
                head.fit_step(Z, yb)
        self._models["scratch"] = (ae_scratch, head)

    # ── Baseline: sklearn models ───────────────────────────────────────────────
    def fit_baselines(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info("Training: Baselines (RF, LR) …")
        sc = StandardScaler().fit(X_train)
        Xs = sc.transform(X_train)
        self._scalers["baseline"] = sc

        rf = RandomForestClassifier(n_estimators=200, random_state=self.seed, n_jobs=1)
        rf.fit(X_train, y_train)
        self._models["baseline_rf"] = rf

        lr = LogisticRegression(max_iter=1000, random_state=self.seed, C=0.1)
        lr.fit(Xs, y_train)
        self._models["baseline_lr"] = lr

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray, strategy: str) -> np.ndarray:
        if strategy == "transfer_frozen":
            Z = self.ae.encode(X)
            return self._models["transfer_frozen"].predict_proba(Z)
        elif strategy == "transfer_finetune":
            ae_ft, head = self._models["transfer_finetune"]
            Z = ae_ft.encode(X)
            return head.predict_proba(Z)
        elif strategy == "scratch":
            ae_s, head = self._models["scratch"]
            Z = ae_s.encode(X)
            return head.predict_proba(Z)
        elif strategy == "baseline_rf":
            return self._models["baseline_rf"].predict_proba(X)[:, 1]
        elif strategy == "baseline_lr":
            Xs = self._scalers["baseline"].transform(X)
            return self._models["baseline_lr"].predict_proba(Xs)[:, 1]
        raise ValueError(f"Unknown strategy: {strategy}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, dict]:
        strategies = [k for k in ["transfer_frozen", "transfer_finetune", "scratch",
                                   "baseline_rf", "baseline_lr"]
                      if k in self._models]
        results = {}
        for strat in strategies:
            proba = self.predict_proba(X_test, strat)
            pred  = (proba >= 0.5).astype(int)
            fpr, tpr, _ = roc_curve(y_test, proba)
            results[strat] = {
                "accuracy":  round(accuracy_score(y_test, pred), 4),
                "auc":       round(roc_auc_score(y_test, proba), 4),
                "f1":        round(f1_score(y_test, pred), 4),
                "fpr":       fpr,
                "tpr":       tpr,
                "confusion": confusion_matrix(y_test, pred),
                "proba":     proba,
            }
            logger.info("  %-22s  acc=%.3f  auc=%.3f  f1=%.3f",
                        strat, results[strat]["accuracy"],
                        results[strat]["auc"], results[strat]["f1"])
        self.results = results
        return results

    def learning_curve(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test:  np.ndarray, y_test:  np.ndarray,
        train_sizes: list[int] | None = None,
    ) -> dict[str, list[float]]:
        """AUC vs training set size for transfer_frozen vs scratch."""
        if train_sizes is None:
            train_sizes = [10, 20, 30, 40, 60, 80, 100]
        train_sizes = [s for s in train_sizes if s <= len(X_train)]

        auc_transfer, auc_scratch = [], []
        rng = np.random.default_rng(self.seed + 10)

        for n in train_sizes:
            idx = rng.choice(len(X_train), n, replace=False)
            Xn, yn = X_train[idx], y_train[idx]
            if yn.sum() == 0 or yn.sum() == len(yn):
                auc_transfer.append(np.nan)
                auc_scratch.append(np.nan)
                continue

            # Transfer frozen
            Z = self.ae.encode(Xn)
            head_t = _MLPHead(Z.shape[1], hidden=32, lr=5e-4, seed=0)
            r2 = np.random.default_rng(0)
            for _ in range(60):
                ii = r2.permutation(len(Z))
                for s in range(0, len(Z), 16):
                    b = ii[s:s+16]
                    head_t.fit_step(Z[b], yn[b])
            p_t = head_t.predict_proba(self.ae.encode(X_test))
            auc_transfer.append(roc_auc_score(y_test, p_t))

            # Scratch
            ae_sc = MicrobiomeAutoencoder(
                input_dim=X_train.shape[1],
                hidden_dims=self.ae.hidden_dims,
                latent_dim=self.ae.latent_dim, seed=99)
            head_s = _MLPHead(self.ae.latent_dim, hidden=32, lr=5e-4, seed=0)
            r3 = np.random.default_rng(1)
            for _ in range(60):
                ii = r3.permutation(len(Xn))
                for s in range(0, len(Xn), 16):
                    b = ii[s:s+16]
                    head_s.fit_step(ae_sc.encode(Xn[b]), yn[b])
            p_s = head_s.predict_proba(ae_sc.encode(X_test))
            auc_scratch.append(roc_auc_score(y_test, p_s))

        return {
            "train_sizes":    train_sizes,
            "auc_transfer":   auc_transfer,
            "auc_scratch":    auc_scratch,
        }
