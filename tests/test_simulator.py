"""Tests for src/data/simulator.py"""
import numpy as np
import pandas as pd
import pytest
from src.data.simulator import (
    simulate_source_domain, simulate_target_domain,
    OTU_NAMES, N_OTUS, IBD_ENRICHED, IBD_DEPLETED, _clr_transform,
)


class TestSimulator:
    def test_source_shape(self):
        X = simulate_source_domain(n_samples=50, seed=0)
        assert X.shape == (50, N_OTUS)

    def test_source_reproducible(self):
        X1 = simulate_source_domain(n_samples=20, seed=7)
        X2 = simulate_source_domain(n_samples=20, seed=7)
        pd.testing.assert_frame_equal(X1, X2)

    def test_source_no_nan(self):
        X = simulate_source_domain(n_samples=30, seed=1)
        assert not np.isnan(X.values).any()

    def test_target_shape(self):
        X, y = simulate_target_domain(n_samples=40, seed=0)
        assert X.shape == (40, N_OTUS)
        assert len(y) == 40

    def test_target_labels(self):
        X, y = simulate_target_domain(n_samples=20, n_ibd=8, seed=0)
        assert (y == 1).sum() == 8
        assert (y == 0).sum() == 12

    def test_ibd_enriched_higher(self):
        X, y = simulate_target_domain(n_samples=100, seed=42)
        ctrl = X.values[y.values == 0]
        ibd  = X.values[y.values == 1]
        genes = [g for g in IBD_ENRICHED if g in X.columns]
        assert len(genes) > 0
        assert ibd[:, [X.columns.get_loc(g) for g in genes]].mean() > \
               ctrl[:, [X.columns.get_loc(g) for g in genes]].mean()

    def test_ibd_depleted_lower(self):
        X, y = simulate_target_domain(n_samples=100, seed=42)
        ctrl = X.values[y.values == 0]
        ibd  = X.values[y.values == 1]
        genes = [g for g in IBD_DEPLETED if g in X.columns]
        assert len(genes) > 0
        assert ibd[:, [X.columns.get_loc(g) for g in genes]].mean() < \
               ctrl[:, [X.columns.get_loc(g) for g in genes]].mean()

    def test_clr_transform_zero_sum(self):
        X = np.abs(np.random.default_rng(0).normal(size=(10, 20))) + 0.01
        clr = _clr_transform(X)
        row_means = clr.mean(axis=1)
        assert np.allclose(row_means, 0, atol=1e-10)

    def test_otu_names_length(self):
        assert len(OTU_NAMES) == N_OTUS == 200
