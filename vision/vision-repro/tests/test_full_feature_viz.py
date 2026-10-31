import random
import torch
import numpy as np
from vision_repro.five_variants import channel_scores, path_vector, diversity_penalty
from vision_repro.full_feature_viz import (scores_for, paths_for, grouped_diversity,
    batch_augment, previous_layer, BatchRedirectReLU, RedirectMask, config_A)
from vision_repro.optimize import augment
from vision_repro.full_path_anchors import topk_stable


def test_independent_channel_scores_and_paths():
    a = torch.rand(3, 4, 7, 7, requires_grad=True)
    b = torch.rand(3, 5, 5, 5, requires_grad=True)
    channels = torch.tensor([0, 2, 4])
    assert torch.allclose(scores_for(b, channels), torch.cat([channel_scores(b[i:i+1], int(c)) for i,c in enumerate(channels)]))
    assert torch.allclose(paths_for(a,b,channels), torch.cat([path_vector(a[i:i+1], b[i:i+1], int(c)) for i,c in enumerate(channels)]), atol=1e-6)


def test_diversity_never_compares_different_neurons():
    x = torch.rand(8, 6, 5, 5, requires_grad=True)
    actual = grouped_diversity(x, 2, 4)
    expected = torch.stack([diversity_penalty(x[:4]), diversity_penalty(x[4:])])
    assert torch.allclose(actual, expected)
    actual[0].backward()
    assert x.grad[:4].abs().sum() > 0 and x.grad[4:].abs().sum() == 0


def test_augment_stream_matches_standalone_neuron():
    cfg = config_A()
    x = torch.rand(4, 3, 12, 12)
    rngs = [random.Random(0), random.Random(0)]
    got = batch_augment(x, cfg, rngs, 2)
    expected = []
    for group in range(2):
        random.seed(0)
        expected.extend(augment(y[None],cfg) for y in x[group*2:group*2+2])
    assert torch.equal(got, torch.cat(expected))


def test_relu_releases_only_selected_independent_sample():
    mask = RedirectMask()
    mask.enabled = torch.tensor([True, False])
    x = torch.full((2, 3), -1., requires_grad=True)
    (-BatchRedirectReLU(mask)(x).sum()).backward()
    assert (x.grad[0] == -1).all() and (x.grad[1] == 0).all()


def test_schedule_does_not_shrink_with_run_length():
    cfg = config_A(1024)
    assert cfg['schedule_steps'] == 768
    assert cfg['steps'] == 1024 and cfg['fft_decay_power'] == 1.5
    assert cfg['logit_chroma_gain'] == 2 and cfg['mean_color_weight'] == 20000
    assert previous_layer('mixed4c') == 'mixed4b'
    assert previous_layer('conv2d0') == 'input_rgb'


def test_anchor_topk_stable_with_ties():
    values=np.array([3.,2.,3.,0.,-2.,-2.,2.])
    assert topk_stable(values,3).tolist()==[0,2,1]
    assert topk_stable(values,3,negative=True).tolist()==[4,5,3]
    assert topk_stable(np.zeros(100),60).tolist()==list(range(60))
