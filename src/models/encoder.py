"""
MLP Encoder for Transfer Learning on Microbiome Data.

Architecture
-----------
Source domain pretraining:
  Input (n_genera) → [256] → [128] → [64] → Output (n_diseases)

Transfer strategies compared:
  1. frozen_encoder    — pretrained [256→128→64] encoder, new linear head
  2. fine_tuned_encoder — pretrained encoder, all weights updated on target
  3. scratch_mlp       — same architecture trained from scratch on target
  4. random_forest     — traditional ML baseline (no transfer)

Implementation
--------------
Uses sklearn's MLPClassifier as the neural backbone.
Transfer is implemented by:
  (a) Extracting hidden layer activations from pretrained MLP
  (b) Training a lightweight head (LogisticRegression) on those embeddings
  This mimics "frozen encoder + new head" in PyTorch transfer learning.

For fine-tuning: re-initialise the final MLP layer using pretrained
weights as starting point (warm-start with transferred hidden weights).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MicrobiomeEncoder:
    """
    Pretrained MLP encoder for microbiome feature extraction.

    The encoder is pretrained on a multi-disease source domain,
    then transferred to a new target classification task.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (256, 128, 64),
        max_iter: int = 500,
        learning_rate_init: float = 0.001,
        alpha: float = 1e-4,
        random_state: int = 42,
    ):
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.random_state = random_state
        self._mlp: Optional[MLPClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self.is_fitted = False

    # ─── Pretraining ────────────────────────────────────────────────────────

    def pretrain(self, X: np.ndarray, y: np.ndarray) -> "MicrobiomeEncoder":
        """
        Pretrain the encoder on source-domain data (multiple diseases).

        Parameters
        ----------
        X : (n_samples, n_features) CLR-transformed microbiome features.
        y : (n_samples,) multi-class disease labels.
        """
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate_init,
            alpha=self.alpha,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=False,
        )
        self._mlp.fit(X_scaled, y)
        self.is_fitted = True
        n_iter = self._mlp.n_iter_
        loss = self._mlp.best_loss_
        logger.info(
            "Encoder pretrained: %d iters | best loss=%.4f | classes=%d",
            n_iter, loss, len(np.unique(y)),
        )
        return self

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Extract the last hidden layer activations (64-dim embedding).

        This is the "frozen encoder" output used as features
        for the target-domain head.
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be pretrained before calling get_embeddings()")
        X_scaled = self._scaler.transform(X)
        # Forward pass through all hidden layers
        activations = X_scaled
        for i, (W, b) in enumerate(zip(self._mlp.coefs_[:-1], self._mlp.intercepts_[:-1])):
            activations = np.maximum(0, activations @ W + b)   # ReLU
        return activations   # shape: (n_samples, hidden_layers[-1])

    def pretrained_weights(self) -> tuple[list, list]:
        """Return copies of the encoder weight matrices and biases."""
        if not self.is_fitted:
            raise RuntimeError("Encoder must be pretrained first")
        return (
            [W.copy() for W in self._mlp.coefs_],
            [b.copy() for b in self._mlp.intercepts_],
        )


# ─── Transfer Strategies ──────────────────────────────────────────────────────

class FrozenEncoderClassifier:
    """
    Strategy 1: Frozen Encoder + New Head.

    Uses pretrained encoder embeddings as features,
    trains a logistic regression head on top.
    No encoder weights are updated.
    """

    def __init__(self, encoder: MicrobiomeEncoder):
        self.encoder = encoder
        self.head = LogisticRegression(
            max_iter=1000, random_state=42, C=1.0, solver="lbfgs"
        )
        self.name = "frozen_encoder"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FrozenEncoderClassifier":
        embeddings = self.encoder.get_embeddings(X)
        self.head.fit(embeddings, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.head.predict(self.encoder.get_embeddings(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.head.predict_proba(self.encoder.get_embeddings(X))


class FineTunedEncoderClassifier:
    """
    Strategy 2: Fine-Tuned Encoder.

    Initialises the MLP with pretrained encoder weights (warm start),
    then re-trains all weights on target domain data.
    This simulates PyTorch-style fine-tuning with a smaller learning rate.
    """

    def __init__(self, encoder: MicrobiomeEncoder, n_iter: int = 200):
        self.encoder = encoder
        self.n_iter = n_iter
        self.name = "fine_tuned_encoder"
        self._mlp: Optional[MLPClassifier] = None
        self._scaler: Optional[StandardScaler] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FineTunedEncoderClassifier":
        self._scaler = deepcopy(self.encoder._scaler)
        X_scaled = self._scaler.transform(X)

        pretrained_W, pretrained_b = self.encoder.pretrained_weights()

        # Build new MLP and warm-start with pretrained weights
        self._mlp = MLPClassifier(
            hidden_layer_sizes=self.encoder.hidden_layers,
            activation="relu",
            solver="adam",
            max_iter=self.n_iter,
            learning_rate_init=self.encoder.learning_rate_init * 0.1,  # smaller LR
            alpha=self.encoder.alpha,
            random_state=42,
            warm_start=False,
            verbose=False,
        )
        # Do one fit call to initialise internal structure
        self._mlp.fit(X_scaled, y)

        # Overwrite hidden-layer weights with pretrained values
        # (keep the final output layer as re-initialised for target classes)
        n_hidden = len(self.encoder.hidden_layers)
        for layer_i in range(n_hidden):
            if layer_i < len(pretrained_W) - 1:
                # Adapt weight dimensions if class count differs
                W_pre = pretrained_W[layer_i]
                W_cur = self._mlp.coefs_[layer_i]
                rows = min(W_pre.shape[0], W_cur.shape[0])
                cols = min(W_pre.shape[1], W_cur.shape[1])
                self._mlp.coefs_[layer_i][:rows, :cols] = W_pre[:rows, :cols]
                self._mlp.intercepts_[layer_i][:cols] = pretrained_b[layer_i][:cols]

        # Fine-tune with warm start
        self._mlp.set_params(warm_start=True, max_iter=self.n_iter)
        self._mlp.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._mlp.predict(self._scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._mlp.predict_proba(self._scaler.transform(X))


class ScratchMLPClassifier:
    """
    Strategy 3: MLP Trained from Scratch.

    Same architecture as the encoder, trained only on target domain.
    No pretrained weights — baseline for transfer learning comparison.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (256, 128, 64),
        max_iter: int = 500,
    ):
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.name = "scratch_mlp"
        self._pipeline: Optional[Pipeline] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchMLPClassifier":
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=self.hidden_layers,
                activation="relu",
                solver="adam",
                max_iter=self.max_iter,
                learning_rate_init=0.001,
                alpha=1e-4,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                verbose=False,
            )),
        ])
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict_proba(X)


class RandomForestBaseline:
    """
    Strategy 4: Random Forest Baseline.

    Traditional ML without any transfer. Uses raw CLR features.
    Gold standard comparison for microbiome ML.
    """

    def __init__(self, n_estimators: int = 300):
        self.n_estimators = n_estimators
        self.name = "random_forest"
        self._pipeline: Optional[Pipeline] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )),
        ])
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict_proba(X)
