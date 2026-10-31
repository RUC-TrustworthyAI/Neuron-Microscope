"""Local color metrics on rendered RGB, without modifying model-input pixels."""
import torch
import torch.nn.functional as F


def neighbor_rgb_loss(image):
    """Equal-weight horizontal/vertical edge MSE; each undirected edge once."""
    if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 2:
        raise ValueError('Expected Bx3xHxW RGB with H,W >= 2')
    horizontal = image[:, :, :, 1:] - image[:, :, :, :-1]
    vertical = image[:, :, 1:, :] - image[:, :, :-1, :]
    return (horizontal.square().mean() + vertical.square().mean()) / 2


def patch_pair_loss(image, size=4):
    """Mean squared difference of ALL unordered pairs in sliding valid patches.

    For n pixels, average over n*(n-1)/2 pairs equals 2*unbiased variance.
    Average equally over batch, color coordinates and stride-one windows. No
    padding or wraparound. Includes diagonal and separated pixels, not merely
    repeated one-pixel edges. Centered variance avoids E[x^2]-E[x]^2 cancellation.
    """
    if type(size) is not int or size < 2:
        raise ValueError('patch size must be an integer >= 2')
    if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < size:
        raise ValueError('Expected Bx3xHxW with H,W >= patch size')
    patches = F.unfold(image, kernel_size=size, stride=1)
    patches = patches.reshape(image.shape[0], 3, size * size, -1)
    return 2 * patches.var(dim=2, unbiased=True).mean()


def validate_local_color(config):
    if config.get('neighbor_color_space', 'rgb') not in ('rgb', 'pca_whitened'):
        raise ValueError('Unknown neighbor_color_space')
    if config.get('neighbor_neighborhood', 'edges') not in ('edges', 'patch'):
        raise ValueError('Unknown neighbor_neighborhood')
    size = config.get('neighbor_patch_size', 4)
    if type(size) is not int or size < 2 or size > config['size']:
        raise ValueError('neighbor_patch_size must be an integer in [2, image size]')


class LocalColorLoss(torch.nn.Module):
    def __init__(self, config, fitted_matrix):
        super().__init__()
        validate_local_color(config)
        self.space = config.get('neighbor_color_space', 'rgb')
        self.neighborhood = config.get('neighbor_neighborhood', 'edges')
        self.patch_size = config.get('neighbor_patch_size', 4)
        matrix = torch.as_tensor(fitted_matrix, dtype=torch.float64)
        if matrix.shape != (3, 3) or not torch.isfinite(matrix).all():
            raise ValueError('Expected finite 3x3 fitted color matrix')
        transform = torch.eye(3, dtype=torch.float64)
        divisor = 1.
        if self.space == 'pca_whitened':
            # Fitted A maps whitened latent coordinates to RGB. A pure rotation
            # would leave L2 invariant. A^-1 also rescales by inverse PCA std.
            transform = torch.linalg.inv(matrix)
            # Equal expected loss for isotropic RGB noise: trace(W.T W) = 3.
            # Thus this changes relative color-axis weights, not overall scale.
            divisor = float((transform.square().sum() / 3).sqrt())
            transform = transform / divisor
        self.register_buffer('transform', transform.float())
        self.description = {
            'color_space': self.space, 'neighborhood': self.neighborhood,
            'patch_size': self.patch_size if self.neighborhood == 'patch' else None,
            'coordinate_transform': transform.tolist(),
            'inverse_matrix_normalization_divisor': divisor,
            'normalization': 'trace(W.T W)=3; equal expectation to RGB for isotropic RGB noise',
            'domain': 'rendered unaugmented RGB after annealed/final blur; not logits',
            'spatial_definition': ('mean over all 120 unordered pairs in every valid stride-one 4x4 patch'
                                   if self.neighborhood == 'patch' and self.patch_size == 4 else
                                   'all unordered pairs per valid stride-one patch' if self.neighborhood == 'patch' else
                                   '0.5*(mean horizontal edge squared difference + mean vertical edge squared difference)'),
            'offset': 'No mean subtraction needed: constant color offsets cancel in differences',
        }

    def forward(self, image):
        coordinates = image
        if self.space != 'rgb':
            coordinates = torch.einsum('ij,bjhw->bihw', self.transform.to(image), image)
        if self.neighborhood == 'patch':
            return patch_pair_loss(coordinates, self.patch_size)
        return neighbor_rgb_loss(coordinates)
