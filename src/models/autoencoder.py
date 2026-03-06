"""
Microbiome Autoencoder — pure NumPy implementation.

Architecture:
    Encoder: input(200) → Dense(128, ReLU) → Dense(64, ReLU) → latent(32)
    Decoder: latent(32) → Dense(64, ReLU) → Dense(128, ReLU) → output(200)

Trained on the large unlabeled source domain (HMP-like) with MSE loss.
The encoder weights are later reused as pretrained feature extractor.
"""
from __future__ import annotations
import logging
import numpy as np
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


def relu(x):       return np.maximum(0, x)
def relu_grad(x):  return (x > 0).astype(float)

def he_init(fan_in, fan_out, rng):
    return rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))


class MicrobiomeAutoencoder:
    def __init__(self, input_dim=200, hidden_dims=(128, 64),
                 latent_dim=32, lr=1e-3, seed=0):
        self.input_dim   = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim  = latent_dim
        self.lr          = lr
        self.seed        = seed
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []
        self._mu  = None
        self._std = None
        self._init_weights()

    def _init_weights(self):
        rng  = np.random.default_rng(self.seed)
        dims = [self.input_dim] + list(self.hidden_dims) + [self.latent_dim]
        self.enc_W = [he_init(dims[i], dims[i+1], rng) for i in range(len(dims)-1)]
        self.enc_b = [np.zeros(dims[i+1]) for i in range(len(dims)-1)]
        dec_dims = list(reversed(dims))
        self.dec_W = [he_init(dec_dims[i], dec_dims[i+1], rng) for i in range(len(dec_dims)-1)]
        self.dec_b = [np.zeros(dec_dims[i+1]) for i in range(len(dec_dims)-1)]

    def _normalise(self, X):
        if self._mu is None: return X
        return (X - self._mu) / self._std

    def encode(self, X):
        h = self._normalise(X)
        for W, b in zip(self.enc_W, self.enc_b):
            h = relu(h @ W + b)
        return h

    def decode(self, Z):
        h = Z
        for i, (W, b) in enumerate(zip(self.dec_W, self.dec_b)):
            h = h @ W + b
            if i < len(self.dec_W) - 1:
                h = relu(h)
        return h

    def forward(self, X):
        Z = self.encode(X)
        return Z, self.decode(Z)

    def _forward_cache(self, X_norm):
        enc_pre, enc_act = [], [X_norm]
        h = X_norm
        for W, b in zip(self.enc_W, self.enc_b):
            pre = h @ W + b
            enc_pre.append(pre); h = relu(pre); enc_act.append(h)
        dec_pre, dec_act = [], [h]
        for i, (W, b) in enumerate(zip(self.dec_W, self.dec_b)):
            pre = h @ W + b
            dec_pre.append(pre)
            h = pre if i == len(self.dec_W)-1 else relu(pre)
            dec_act.append(h)
        return enc_pre, enc_act, dec_pre, dec_act

    def _clip(self, grads, clip=1.0):
        norm = max(np.sqrt(sum(np.sum(g**2) for g in grads)), 1e-10)
        if norm > clip:
            grads = [g * (clip / norm) for g in grads]
        return grads

    def _backward(self, X_norm, enc_pre, enc_act, dec_pre, dec_act, clip=1.0):
        n = X_norm.shape[0]
        dL = 2 * (dec_act[-1] - X_norm) / n
        dW_dec, db_dec, delta = [], [], dL
        for i in reversed(range(len(self.dec_W))):
            if i < len(self.dec_W)-1:
                delta = delta * relu_grad(dec_pre[i])
            dW_dec.insert(0, dec_act[i].T @ delta)
            db_dec.insert(0, delta.sum(axis=0))
            delta = delta @ self.dec_W[i].T
        dW_enc, db_enc = [], []
        for i in reversed(range(len(self.enc_W))):
            delta = delta * relu_grad(enc_pre[i])
            dW_enc.insert(0, enc_act[i].T @ delta)
            db_enc.insert(0, delta.sum(axis=0))
            delta = delta @ self.enc_W[i].T
        # Clip
        all_grads = dW_enc + dW_dec
        all_grads = self._clip(all_grads, clip)
        dW_enc = all_grads[:len(dW_enc)]
        dW_dec = all_grads[len(dW_enc):]
        return dW_enc, db_enc, dW_dec, db_dec

    def fit(self, X_train, X_val=None, epochs=100, batch_size=64):
        # Normalise for stability
        self._mu  = X_train.mean(axis=0)
        self._std = X_train.std(axis=0) + 1e-8
        Xn_train = (X_train - self._mu) / self._std
        Xn_val   = (X_val - self._mu) / self._std if X_val is not None else None

        rng = np.random.default_rng(self.seed + 1)
        n   = len(Xn_train)

        for epoch in range(epochs):
            idx = rng.permutation(n)
            Xs  = Xn_train[idx]
            ep_loss, n_b = 0.0, 0
            for start in range(0, n, batch_size):
                Xb = Xs[start:start+batch_size]
                ep, ea, dp, da = self._forward_cache(Xb)
                loss = float(np.mean((da[-1] - Xb)**2))
                if np.isnan(loss): continue
                ep_loss += loss; n_b += 1
                dWe, dbe, dWd, dbd = self._backward(Xb, ep, ea, dp, da)
                for i in range(len(self.enc_W)):
                    self.enc_W[i] -= self.lr * dWe[i]
                    self.enc_b[i] -= self.lr * dbe[i]
                for i in range(len(self.dec_W)):
                    self.dec_W[i] -= self.lr * dWd[i]
                    self.dec_b[i] -= self.lr * dbd[i]

            self.train_losses.append(ep_loss / max(n_b, 1))
            if Xn_val is not None:
                _, Xhv = self.forward(X_val)
                Xnv_hat = (Xhv - self._mu) / self._std if False else \
                          self.decode(self.encode(X_val))
                # val loss in normalised space
                Zv = np.copy(Xn_val)
                h = Zv
                for W, b in zip(self.enc_W, self.enc_b):
                    h = relu(h @ W + b)
                hd = h
                for i2, (W, b) in enumerate(zip(self.dec_W, self.dec_b)):
                    hd = hd @ W + b
                    if i2 < len(self.dec_W)-1: hd = relu(hd)
                self.val_losses.append(float(np.mean((hd - Xn_val)**2)))

            if (epoch+1) % 20 == 0:
                vs = f"  val={self.val_losses[-1]:.4f}" if self.val_losses else ""
                logger.info("Epoch %3d/%d  train=%.4f%s",
                            epoch+1, epochs, self.train_losses[-1], vs)
        return self

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved autoencoder → %s", path)

    @staticmethod
    def load(path): return joblib.load(path)
