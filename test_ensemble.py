from datetime import date

import pytest
import config
from ensemble import EnsembleForecaster


def test_compose_sigma_uses_base_when_ensemble_missing():
    ens = EnsembleForecaster(lat=40.0, lon=-73.0)
    out = ens.compose_sigma(sigma_base=1.2, sigma_ens=None)
    assert out == 1.2


def test_compose_sigma_widens_with_ensemble(monkeypatch):
    ens = EnsembleForecaster(lat=40.0, lon=-73.0)
    monkeypatch.setattr(config, "ENSEMBLE_ALPHA", 1.1)
    out = ens.compose_sigma(sigma_base=1.2, sigma_ens=1.5)
    assert out == pytest.approx(1.65)


def test_get_ensemble_sigma_uses_cache(monkeypatch):
    ens = EnsembleForecaster(lat=40.0, lon=-73.0)
    calls = {"n": 0}

    def fake_fetch(model_name, target_date):
        calls["n"] += 1
        return [30.0, 31.0, 32.0]

    monkeypatch.setattr(ens, "_fetch_model_ensemble", fake_fetch)
    monkeypatch.setattr(config, "ENSEMBLE_MODELS", ["m1", "m2"])

    target = date(2026, 2, 11)
    sigma1, n1, _ = ens.get_ensemble_sigma(target)
    sigma2, n2, _ = ens.get_ensemble_sigma(target)

    assert sigma1 is not None
    assert n1 == 6
    assert sigma2 == sigma1
    assert n2 == n1
    assert calls["n"] == 2  # first call only: one per model


def test_get_ensemble_sigma_insufficient_members(monkeypatch):
    ens = EnsembleForecaster(lat=40.0, lon=-73.0)

    monkeypatch.setattr(ens, "_fetch_model_ensemble", lambda model_name, target_date: [33.0])
    monkeypatch.setattr(config, "ENSEMBLE_MODELS", ["m1", "m2"])

    sigma, n_members, values = ens.get_ensemble_sigma(date(2026, 2, 11))

    assert sigma is None
    assert n_members == 2
    assert values == [33.0, 33.0]
