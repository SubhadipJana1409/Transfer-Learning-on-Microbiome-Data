"""Tests for classifier and visualization."""
import numpy as np
import pytest
import pandas as pd
from src.models.autoencoder import MicrobiomeAutoencoder
from src.models.classifier  import TransferClassifier
from src.visualization.plots import (
    fig1_domain_overview, fig2_training_curves, fig5_roc_curves,
    fig7_performance_bar, fig8_learning_curve,
)
from sklearn.metrics import roc_auc_score


@pytest.fixture(scope="module")
def trained_tc():
    rng = np.random.default_rng(0)
    X_src = rng.normal(size=(200, 50))
    ae = MicrobiomeAutoencoder(input_dim=50, hidden_dims=(32,16), latent_dim=8, seed=0)
    ae.fit(X_src, epochs=15, batch_size=32)

    X_tr = rng.normal(size=(60, 50))
    y_tr = np.array([0]*30 + [1]*30)
    X_te = rng.normal(size=(20, 50))
    y_te = np.array([0]*10 + [1]*10)

    tc = TransferClassifier(ae, seed=0)
    tc.fit_transfer_frozen(X_tr, y_tr, epochs=20)
    tc.fit_scratch(X_tr, y_tr, epochs=20)
    tc.fit_baselines(X_tr, y_tr)
    tc.evaluate(X_te, y_te)
    return tc, X_src, X_tr, y_tr, X_te, y_te


class TestTransferClassifier:
    def test_results_keys(self, trained_tc):
        tc = trained_tc[0]
        assert "transfer_frozen" in tc.results
        assert "scratch" in tc.results
        assert "baseline_rf" in tc.results

    def test_auc_range(self, trained_tc):
        tc = trained_tc[0]
        for strat, res in tc.results.items():
            assert 0.0 <= res["auc"] <= 1.0, f"{strat} AUC out of range"

    def test_accuracy_range(self, trained_tc):
        tc = trained_tc[0]
        for strat, res in tc.results.items():
            assert 0.0 <= res["accuracy"] <= 1.0

    def test_confusion_matrix_shape(self, trained_tc):
        tc = trained_tc[0]
        for strat, res in tc.results.items():
            assert res["confusion"].shape == (2, 2)

    def test_transfer_beats_scratch_or_close(self, trained_tc):
        tc = trained_tc[0]
        auc_t = tc.results["transfer_frozen"]["auc"]
        auc_s = tc.results["scratch"]["auc"]
        # Transfer should at minimum not be much worse
        assert auc_t >= auc_s - 0.2, "Transfer is far worse than scratch"

    def test_proba_range(self, trained_tc):
        tc, _, _, _, X_te, _ = trained_tc
        for strat in tc.results:
            p = tc.predict_proba(X_te, strat)
            assert (p >= 0).all() and (p <= 1).all()

    def test_learning_curve_output(self, trained_tc):
        tc, _, X_tr, y_tr, X_te, y_te = trained_tc
        lc = tc.learning_curve(X_tr, y_tr, X_te, y_te, train_sizes=[10, 20, 30])
        assert "train_sizes" in lc
        assert "auc_transfer" in lc
        assert "auc_scratch" in lc
        assert len(lc["auc_transfer"]) == 3

    def test_unknown_strategy_raises(self, trained_tc):
        tc, _, _, _, X_te, _ = trained_tc
        with pytest.raises(ValueError):
            tc.predict_proba(X_te, "nonexistent_strategy")


class TestPlots:
    def test_fig1_creates_file(self, tmp_path):
        rng = np.random.default_rng(0)
        X_src = rng.normal(size=(100, 20))
        X_tgt = rng.normal(size=(30, 20))
        y_tgt = np.array([0]*15 + [1]*15)
        from pathlib import Path
        fig1_domain_overview(X_src, X_tgt, y_tgt, Path(tmp_path))
        assert (tmp_path / "fig1_domain_overview.png").exists()

    def test_fig2_creates_file(self, tmp_path):
        from pathlib import Path
        fig2_training_curves([1.0, 0.9, 0.8], [1.1, 1.0, 0.95], Path(tmp_path))
        assert (tmp_path / "fig2_training_curves.png").exists()

    def test_fig5_creates_file(self, trained_tc, tmp_path):
        from pathlib import Path
        tc = trained_tc[0]
        fig5_roc_curves(tc.results, Path(tmp_path))
        assert (tmp_path / "fig5_roc_curves.png").exists()

    def test_fig7_creates_file(self, trained_tc, tmp_path):
        from pathlib import Path
        tc = trained_tc[0]
        fig7_performance_bar(tc.results, Path(tmp_path))
        assert (tmp_path / "fig7_performance_bar.png").exists()

    def test_fig8_creates_file(self, tmp_path):
        from pathlib import Path
        lc = {"train_sizes": [10,20,30], "auc_transfer": [0.6,0.7,0.8],
              "auc_scratch": [0.5,0.55,0.6]}
        fig8_learning_curve(lc, Path(tmp_path))
        assert (tmp_path / "fig8_learning_curve.png").exists()
