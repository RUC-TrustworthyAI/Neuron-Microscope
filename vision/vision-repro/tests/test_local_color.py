import pytest
import torch
from vision_repro.local_color import LocalColorLoss, neighbor_rgb_loss, patch_pair_loss
from vision_repro.five_variants import run_variant


def test_patch_matches_explicit_pairs_and_gradients():
    torch.manual_seed(2)
    image = torch.rand(2, 3, 5, 6, dtype=torch.float64, requires_grad=True)
    values = []
    for y in range(2):
        for x in range(3):
            patch = image[:, :, y:y+4, x:x+4].flatten(2)
            pairs = [(patch[:, :, i] - patch[:, :, j]).square()
                     for i in range(16) for j in range(i+1, 16)]
            values.append(torch.stack(pairs).mean())
    expected = torch.stack(values).mean()
    actual = patch_pair_loss(image, 4)
    assert torch.allclose(actual, expected, atol=1e-14)
    assert torch.allclose(torch.autograd.grad(actual, image)[0],
                          torch.autograd.grad(expected, image)[0], atol=1e-14)


def test_patch_constant_translation_scaling_and_batch_mean():
    image = torch.full((1, 3, 6, 6), .75, requires_grad=True)
    loss = patch_pair_loss(image)
    assert loss == 0
    loss.backward()
    assert image.grad.count_nonzero() == 0
    x = torch.rand(1, 3, 6, 6, dtype=torch.float64)
    assert torch.allclose(patch_pair_loss(x+3), patch_pair_loss(x))
    assert torch.allclose(patch_pair_loss(x*2), patch_pair_loss(x)*4)
    assert torch.allclose(patch_pair_loss(x.expand(3,-1,-1,-1)), patch_pair_loss(x))


def test_patch_counts_longer_and_diagonal_neighbors():
    ramp = torch.arange(4.).view(1,1,1,4).expand(1,3,4,4)
    assert neighbor_rgb_loss(ramp) == .5
    assert torch.allclose(patch_pair_loss(ramp), torch.tensor(8/3))


def test_pca_rotation_alone_invariant_and_whitening_normalized():
    torch.manual_seed(3)
    rotation = torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))[0]
    matrix = rotation @ torch.diag(torch.tensor([1.,.3,.15],dtype=torch.float64))
    metric = LocalColorLoss({'size':8,'neighbor_color_space':'pca_whitened'}, matrix).double()
    image = torch.randn(1,3,8,8,dtype=torch.float64,requires_grad=True)
    rotated = torch.einsum('ij,bjhw->bihw', rotation.T, image)
    assert torch.allclose(neighbor_rgb_loss(rotated), neighbor_rgb_loss(image))
    assert torch.allclose(metric.transform.square().sum(), torch.tensor(3.,dtype=torch.float64))
    assert not torch.allclose(metric(image), neighbor_rgb_loss(image))
    transform = torch.linalg.inv(matrix)
    transform /= (transform.square().sum()/3).sqrt()
    expected = neighbor_rgb_loss(torch.einsum('ij,bjhw->bihw',transform,image))
    assert torch.allclose(metric(image), expected, atol=1e-6)
    metric(image).backward()
    assert image.grad.isfinite().all() and image.grad.abs().sum() > 0


def test_legacy_identity_and_invalid_config_before_cuda(tmp_path):
    x = torch.randn(1,3,8,8)
    metric = LocalColorLoss({'size':8}, torch.eye(3))
    assert torch.equal(metric(x), neighbor_rgb_loss(x))
    for cfg in [{'neighbor_color_space':'typo'}, {'neighbor_neighborhood':'diagonal'},
                {'neighbor_patch_size':1}, {'neighbor_patch_size':True},
                {'neighbor_patch_size':225}]:
        with pytest.raises(ValueError):
            run_variant(tmp_path/'never_created',447,'positive',None,changes=cfg)
    assert not (tmp_path/'never_created').exists()
