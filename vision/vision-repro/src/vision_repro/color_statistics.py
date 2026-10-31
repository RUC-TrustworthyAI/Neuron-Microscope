"""Small, explicit RGB covariance estimator and PCA color parameterization."""

import hashlib
import json
from pathlib import Path

import numpy as np

LUCID_FACTOR = np.array(
    [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]], dtype=np.float64
)


class RGBMoments:
    def __init__(self):
        self.count = 0
        self.total = np.zeros(3, dtype=np.float64)
        self.cross = np.zeros((3, 3), dtype=np.float64)

    def update(self, pixels):
        x = np.asarray(pixels, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
            raise ValueError("Expected finite Nx3 RGB pixels")
        self.count += len(x)
        self.total += x.sum(axis=0)
        self.cross += x.T @ x

    def result(self):
        if self.count < 2:
            raise ValueError("At least two pixels required")
        mean = self.total / self.count
        covariance = (self.cross - self.count * np.outer(mean, mean)) / (self.count - 1)
        covariance = (covariance + covariance.T) / 2
        return {
            "pixel_count": self.count,
            "mean_rgb": mean.tolist(),
            **factor_from_covariance(covariance),
        }


def factor_from_covariance(covariance):
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        raise ValueError("Expected finite 3x3 covariance")
    if not np.allclose(covariance, covariance.T, atol=1e-12):
        raise ValueError("Covariance must be symmetric")
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if values[-1] <= 0:
        raise ValueError("RGB covariance must be positive definite")
    # Signs of eigenvectors are arbitrary. Align signs (not directions or scales)
    # to Lucid columns so sign flips do not create a trivial seed confound.
    for column in range(3):
        if vectors[:, column] @ LUCID_FACTOR[:, column] < 0:
            vectors[:, column] *= -1
    factor = vectors * np.sqrt(values)[None, :]
    normalization = np.linalg.norm(factor, axis=0).max()
    return {
        "covariance_rgb": covariance.tolist(),
        "eigenvalues_descending": values.tolist(),
        "unit_directions_columns": vectors.tolist(),
        "unnormalized_factor": factor.tolist(),
        "normalization_max_column_norm": float(normalization),
        "normalized_matrix": (factor / normalization).tolist(),
        "relative_column_scales": np.sqrt(values / values[0]).tolist(),
        "factor_definition": "U diag(sqrt(eigenvalues)), columns descending; divide by max column norm; eigenvector signs aligned to Lucid columns only",
    }


def load_fitted_color(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    stats = json.loads(data)
    matrix = np.asarray(stats["normalized_matrix"], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
        raise ValueError("Fitted color matrix must be finite, invertible 3x3")
    if not np.isclose(np.linalg.norm(matrix, axis=0).max(), 1.0, atol=1e-6):
        raise ValueError("Fitted matrix must use max-column-norm normalization")
    return matrix, {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "statistics": stats,
    }
