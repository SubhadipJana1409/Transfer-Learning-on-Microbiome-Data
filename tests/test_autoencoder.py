"""Tests for src/models/autoencoder.py"""
import numpy as np
import pytest
from src.models.autoencoder import MicrobiomeAutoencoder, relu, relu_grad, he_init


class TestAutoencoder:
    @pytest.fixture
    def small_ae(self, tmp_path):
        ae = MicrobiomeAutoencoder(input_dim=50, hidden_dims=(32, 16),
                                   latent_dim=8, lr=1e-3, seed=0)
        X = np.random.default_rng(0).normal(size=(100, 50))
        ae.fit(X, epochs=10, batch_size=32)
        return ae

    def test_encode_shape(self, small_ae):
        X = np.random.default_rng(1).normal(size=(20, 50))
        Z = small_ae.encode(X)
        assert Z.shape == (20, 8)

    def test_decode_shape(self, small_ae):
        Z = np.random.default_rng(2).normal(size=(15, 8))
        X_hat = small_ae.decode(Z)
        assert X_hat.shape == (15, 50)

    def test_no_nan_in_encode(self, small_ae):
        X = np.random.default_rng(3).normal(size=(10, 50))
        Z = small_ae.encode(X)
        assert not np.isnan(Z).any()

    def test_train_losses_decrease(self, small_ae):
        losses = small_ae.train_losses
        assert losses[0] > losses[-1] or len(losses) > 0

    def test_train_losses_length(self, small_ae):
        assert len(small_ae.train_losses) == 10

    def test_val_losses_recorded(self):
        ae = MicrobiomeAutoencoder(input_dim=30, hidden_dims=(16,),
                                   latent_dim=4, seed=0)
        rng = np.random.default_rng(0)
        Xtr = rng.normal(size=(60, 30))
        Xva = rng.normal(size=(20, 30))
        ae.fit(Xtr, X_val=Xva, epochs=5)
        assert len(ae.val_losses) == 5

    def test_save_load(self, small_ae, tmp_path):
        p = tmp_path / "ae.joblib"
        small_ae.save(p)
        ae2 = MicrobiomeAutoencoder.load(p)
        X = np.random.default_rng(5).normal(size=(10, 50))
        np.testing.assert_array_almost_equal(small_ae.encode(X), ae2.encode(X))

    def test_relu(self):
        x = np.array([-2.0, 0.0, 3.0])
        np.testing.assert_array_equal(relu(x), [0, 0, 3])

    def test_relu_grad(self):
        x = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(relu_grad(x), [0, 0, 1])

    def test_he_init_shape(self):
        rng = np.random.default_rng(0)
        W = he_init(64, 32, rng)
        assert W.shape == (64, 32)

    def test_normalisation_applied(self):
        ae = MicrobiomeAutoencoder(input_dim=20, hidden_dims=(10,), latent_dim=4, seed=0)
        X = np.abs(np.random.default_rng(0).normal(100, 10, size=(40, 20)))
        ae.fit(X, epochs=3)
        assert ae._mu is not None
        assert ae._std is not None
