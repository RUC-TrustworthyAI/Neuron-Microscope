import json

import numpy as np
import pytest
import torch

from vision_repro.color_statistics import RGBMoments, factor_from_covariance, load_fitted_color
from vision_repro.positive import ScheduledFFT


def test_streaming_moments_and_covariance_factor():
    rng = np.random.default_rng(12)
    values = rng.random((1003, 3)) @ np.array([[1.0, 0.2, 0.3], [0.1, 1.0, 0.3], [0.2, 0.4, 1.0]])
    moments = RGBMoments()
    for chunk in np.array_split(values, 17):
        moments.update(chunk)
    result = moments.result()
    assert result["pixel_count"] == len(values)
    assert np.allclose(result["mean_rgb"], values.mean(axis=0))
    covariance = np.cov(values.T, ddof=1)
    assert np.allclose(result["covariance_rgb"], covariance)
    matrix = np.array(result["normalized_matrix"])
    assert np.allclose(matrix @ matrix.T, covariance / result["eigenvalues_descending"][0])
    directions = np.array(result["unit_directions_columns"])
    assert np.allclose(directions.T @ directions, np.eye(3))


def test_fitted_file_hash_and_parameterization_round_trip(tmp_path):
    result = factor_from_covariance(
        np.array([[0.1, 0.07, 0.05], [0.07, 0.12, 0.06], [0.05, 0.06, 0.11]])
    )
    path = tmp_path / "statistics.json"
    path.write_text(json.dumps(result))
    matrix, provenance = load_fitted_color(path)
    assert len(provenance["sha256"]) == 64
    parameter = ScheduledFFT(33, 0.04, "cpu", "fitted", 4, matrix)
    image = torch.rand(1, 3, 33, 33).clamp(0.01, 0.99)
    parameter.initialize_rgb(image)
    assert torch.allclose(parameter(1, 0), image, atol=1e-5)
    parameter(0.3, 1).mean().backward()
    assert torch.isfinite(parameter.coefficients.grad).all()
    assert parameter.coefficients.grad.norm() > 0


@pytest.mark.parametrize("bad", [np.zeros((3, 3)), np.eye(2), np.diag([1.0, 1.0, -1.0])])
def test_bad_covariance_rejected(bad):
    with pytest.raises(ValueError):
        factor_from_covariance(bad)


def test_same_base_noise_but_distinct_color_maps():
    matrix = factor_from_covariance(np.diag([0.09, 0.04, 0.01]))["normalized_matrix"]
    torch.manual_seed(4)
    old = ScheduledFFT(33, 0.04, "cpu", "lucid_svd", 4)
    torch.manual_seed(4)
    new = ScheduledFFT(33, 0.04, "cpu", "fitted", 4, matrix)
    assert old.base_noise_sha256 == new.base_noise_sha256
    assert not torch.allclose(old(), new())
