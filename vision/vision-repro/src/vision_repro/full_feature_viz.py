"""Batched, all-layer A recipe. Independent objectives; diversity stays within neuron.

All variants use the selected A image parameterization. Path/diversity objectives
remain the existing public-description-inspired definitions, generalized to the
previous catalog layer (input RGB for conv2d0). No optimizer/model downloads.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .all_neuron_examples import LAYERS
from .color_statistics import load_fitted_color
from .five_variants import (BASE_CONFIG, BRANCHES, SLOTS, BatchFFT, gaussian_weights,
                            save_batch, sha256, write_json)
from .models import PROJECT_ROOT, load_inceptionv1, preprocess
from .rgb_explore import optimization_schedule, periphery_mask, render

DEFAULT_COLOR = Path(__file__).parent / 'defaults/imagenet_color.json'
DEFAULT_SCAN = PROJECT_ROOT / 'outputs/imagenet_scan'


def config_A(steps=1024):
    suite = json.loads((Path(__file__).parent / 'defaults/A.json').read_text())
    return {**BASE_CONFIG, **suite[0][2], 'steps': steps, 'schedule_steps': 768,
            'save_steps': sorted({s for s in [256, 512, 768, steps] if s <= steps}),
            'history_every': 16, 'seed': 0}


def previous_layer(layer):
    names = [x[0] for x in LAYERS]
    i = names.index(layer)
    return names[i - 1] if i else 'input_rgb'


class EndLayer(Exception):
    pass


class LayerCapture:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.previous = previous_layer(layer)
        self.handles = []
        self.values = {}
        modules = dict(model.named_modules())
        self.branches = ([f'{layer}_pre_relu_conv'] if layer.startswith('conv') else
                         [f'{layer}_{b}_pre_relu_conv' for b in BRANCHES])
        self.mapping = [{'node': name, 'channel': ch} for name in self.branches
                        for ch in range(modules[name].out_channels)]
        names = [*self.branches, layer]
        if self.previous != 'input_rgb':
            names.insert(0, self.previous)
        for name in names:
            def hook(_module, _inputs, value, name=name):
                self.values[name] = value
                if name == layer:
                    raise EndLayer()
            self.handles.append(modules[name].register_forward_hook(hook))

    def __call__(self, image):
        self.values = {'input_rgb': image}
        try:
            self.model(preprocess(image))
        except EndLayer:
            pass
        parts = [self.values[n] for n in self.branches]
        pre = parts[0] if len(parts) == 1 else torch.cat(parts, 1)
        return self.values[self.previous], self.values[self.layer], pre

    def close(self):
        for h in self.handles:
            h.remove()
        self.values = {}


class RedirectMask:
    enabled = None


class _BatchRedirect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x)
        ctx.mask = mask
        return x.clamp_min(0)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        normal = torch.where(x < 0, 0., grad)
        if ctx.mask.enabled is None:
            return normal, None
        blocked = (normal.flatten(1).norm(dim=1) == 0) & ctx.mask.enabled
        redirected = torch.where((x < 0) & (grad > 0), 0., grad)
        return torch.where(blocked.reshape((-1,) + (1,) * (x.ndim - 1)), redirected, normal), None


class BatchRedirectReLU(torch.nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        return _BatchRedirect.apply(x, self.mask)


def install_batch_relu(model, mask, extra_types=()):
    for name, module in list(model.named_children()):
        if isinstance(module, (torch.nn.ReLU, *extra_types)):
            setattr(model, name, BatchRedirectReLU(mask))
        else:
            install_batch_relu(module, mask, extra_types)


def scores_for(field, channels, sigma=.12):
    selected = field[torch.arange(len(channels), device=field.device), channels]
    return (selected * gaussian_weights(field, sigma)).sum((-2, -1))


def paths_for(previous, post, channels, sigma=.12, centers=None):
    a = (previous * gaussian_weights(previous, sigma, centers)[:, None]).sum((-2, -1))
    b = (post * gaussian_weights(post, sigma, centers)[:, None]).sum((-2, -1))
    mask = torch.ones_like(b).scatter_(1, channels[:, None], 0)
    return F.normalize(torch.cat((F.normalize(a, dim=1), F.normalize(b * mask, dim=1)), 1), dim=1)


def grouped_diversity(previous, groups, members, sigma=.12):
    if members == 1:
        return previous.new_zeros(groups)
    x = previous.flatten(2) * gaussian_weights(previous, sigma).flatten(1)[:, None].sqrt()
    gram = torch.bmm(x, x.transpose(1, 2)).flatten(1)
    unit = F.normalize(gram, dim=1).reshape(groups, members, -1)
    cosine = torch.bmm(unit, unit.transpose(1, 2))
    mask = ~torch.eye(members, dtype=torch.bool, device=previous.device)
    return cosine[:, mask].mean(1)


class VectorFFT(torch.nn.Module):
    def __init__(self, groups, members, config, matrix):
        super().__init__()
        torch.manual_seed(config['seed'])
        template = BatchFFT(members, config, matrix)
        self.size = config['size']
        self.seed_hashes = [p.base_noise_sha256 for p in template.images]
        self.coefficients = torch.nn.Parameter(torch.cat([p.coefficients.detach() for p in template.images]).repeat(groups, 1, 1, 1, 1))
        for key in ['scale', 'radius', 'color_matrix']:
            self.register_buffer(key, getattr(template.images[0], key).clone())

    def rgb_logits(self, cutoff):
        phase = ((self.radius - cutoff * .8) / (cutoff * .2)).clamp(0, 1)
        mask = (1 + torch.cos(math.pi * phase)) / 2
        spectrum = torch.view_as_complex(self.coefficients.contiguous()) * self.scale * mask
        latent = torch.fft.irfft2(spectrum, s=(self.size, self.size), norm='ortho') / 4.
        return torch.einsum('ij,bjhw->bihw', self.color_matrix, latent)


def batch_augment(image, config, rngs, members):
    transforms = []
    for rng in rngs:
        for _ in range(members):
            angle = math.radians(rng.uniform(-config['rotation'], config['rotation']))
            scale = rng.uniform(config['scale_min'], config['scale_max'])
            j = config['jitter']
            dx, dy = rng.randint(-j, j), rng.randint(-j, j)
            transforms.append([[math.cos(angle)/scale, -math.sin(angle)/scale, 2*dx/image.shape[-1]],
                               [math.sin(angle)/scale, math.cos(angle)/scale, 2*dy/image.shape[-2]]])
    theta = image.new_tensor(transforms)
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    return F.grid_sample(image, grid, padding_mode='reflection', align_corners=False)


def atomic_torch(path, value):
    temporary = path.with_suffix('.pending')
    torch.save(value, temporary)
    temporary.replace(path)


class BatchRunner:
    def __init__(self, fitted=DEFAULT_COLOR):
        torch.set_num_threads(4)
        self.source_hashes = {name: sha256(Path(__file__).with_name(name)) for name in
                              ['full_feature_viz.py', 'five_variants.py', 'positive.py',
                               'rgb_explore.py', 'models.py', 'color_statistics.py']}
        self.model, self.provenance = load_inceptionv1()
        from lucent.modelzoo.inceptionv1.helper_layers import ReluLayer
        self.redirect = RedirectMask()
        install_batch_relu(self.model, self.redirect, (ReluLayer,))
        self.matrix, self.color_provenance = load_fitted_color(fitted)

    def run(self, output, layer, channels, slot, config, anchors=None, resume=True):
        config = {k: v for k, v in config.items() if k not in ('channel', 'slot')}
        config.update(layer=layer, slot=slot)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        identity = {'layer': layer, 'channels': channels, 'slot': slot, 'config': config}
        if (output / 'job.json').exists():
            if json.loads((output/'job.json').read_text()) != identity:
                raise ValueError('Configuration differs from the existing batch configuration')
            if (output/'complete.json').exists():
                return json.loads((output/'complete.json').read_text())
        else:
            write_json(output/'job.json', identity)
        members = 4 if slot == 'diversity' else 1
        groups = len(channels)
        negative = slot in ('negative', 'path_negative')
        guided = slot in ('path_positive', 'path_negative')
        capture = LayerCapture(self.model, layer)
        parameter = VectorFFT(groups, members, config, self.matrix)
        ch = torch.tensor(channels, device='cuda').repeat_interleave(members)
        optimizer = torch.optim.Adam(parameter.parameters(), lr=config['lr'])
        rngs = [random.Random(config['seed']) for _ in channels]
        active = torch.ones(groups, dtype=torch.bool, device='cuda') if not negative else torch.zeros(groups, dtype=torch.bool, device='cuda')
        first = torch.full((groups,), -1, device='cuda', dtype=torch.long)
        vectors = scales = None
        if guided or slot == 'diversity':
            if anchors is None:
                raise ValueError('Real natural anchors required; no synthetic placeholder')
            pole = 'negative' if negative else 'positive'
            entries = [anchors[f'{channel}_{pole}'] for channel in channels]
            scales = torch.tensor([e['score_scale'] for e in entries], device='cuda')
            vectors = torch.stack([e['vector'] for e in entries]).to('cuda').repeat_interleave(members, 0)
        weight = periphery_mask(parameter.coefficients.new_empty((1, 3, config['size'], config['size'])), config)
        history = []
        checkpoint = output/'resume.pt'
        start = 0
        if resume and checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
            parameter.load_state_dict(saved['parameter'])
            optimizer.load_state_dict(saved['optimizer'])
            active, first, start, history = saved['active'], saved['first'], saved['step'], saved['history']
            for rng, state in zip(rngs, saved['rngs']):
                rng.setstate(state)
        torch.cuda.reset_peak_memory_stats()
        evaluation = {}
        def evaluate(label):
            with torch.no_grad():
                image, _ = render(parameter, config, config['final_band'], config['blur_end'])
                previous, post, pre = capture(image)
                if not torch.allclose(pre.relu(), post, atol=1e-4, rtol=1e-5):
                    raise AssertionError(f'Pre/post mismatch: {layer}')
                signed = scores_for(pre, ch, config['gaussian_sigma']).reshape(groups, members)
                positive = scores_for(post, ch, config['gaussian_sigma']).reshape(groups, members)
                path_cos = ((paths_for(previous, post, ch, config['gaussian_sigma'])*vectors).sum(1).reshape(groups, members) if vectors is not None else None)
                div = grouped_diversity(previous, groups, members, config['gaussian_sigma'])
                results = []
                for i, channel in enumerate(channels):
                    folder = output/f'items/{i}'
                    folder.mkdir(parents=True, exist_ok=True)
                    save_batch(image[i*members:(i+1)*members], folder, label)
                    results.append({'pre_relu': signed[i].tolist(), 'post_relu': positive[i].tolist(),
                                    'score': (signed if negative else positive)[i].tolist(),
                                    'path_cosine': path_cos[i].tolist() if path_cos is not None else None,
                                    'diversity_cosine': float(div[i])})
                return results
        try:
            if not start:
                evaluation['initial'] = evaluate('initial')
                write_json(output/'evaluation.json', evaluation)
            else:
                evaluation = json.loads((output/'evaluation.json').read_text())
            torch.cuda.synchronize()
            started = time.monotonic()
            for step in range(start, config['steps']):
                cutoff, sigma, lr = optimization_schedule(step, config)
                optimizer.param_groups[0]['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                self.redirect.enabled = active.repeat_interleave(members)
                image, _ = render(parameter, config, cutoff, sigma)
                previous, post, pre = capture(batch_augment(image, config, rngs, members))
                scores = scores_for(pre if negative else post, ch, config['gaussian_sigma']).reshape(groups, members)
                if not negative:
                    opened = active & (scores.detach() > 0).all(1)
                    first = torch.where(opened, step, first)
                    active = active & ~opened
                    self.redirect.enabled = active.repeat_interleave(members)
                color = (image.mean((2, 3))-image.new_tensor(config['mean_color'])).square().mean(1).reshape(groups, members).mean(1)
                peripheral_avg = (image*weight).sum((2, 3))/weight.sum((2, 3))
                peripheral = (peripheral_avg-image.new_tensor(config['periphery_color'])).square().mean(1).reshape(groups, members).mean(1)
                losses = (1 if negative else -1)*scores.mean(1) + config['mean_color_weight']*color + config['periphery_mean_weight']*peripheral
                path_loss = losses.new_zeros(groups)
                if guided and step < config['path_steps']:
                    path_loss = (1-(paths_for(previous, post, ch, config['gaussian_sigma'])*vectors).sum(1)).reshape(groups, members).mean(1)
                    losses = losses + config['path_weight_ratio']*scales*path_loss
                div = grouped_diversity(previous, groups, members, config['gaussian_sigma']) if members > 1 else losses.new_zeros(groups)
                if members > 1:
                    losses = losses + config['diversity_weight_ratio']*scales*div
                # Sum independent neuron objectives: no batch-size-dependent loss scale.
                losses.sum().backward()
                optimizer.step()
                if step % config['history_every'] == 0 or step+1 == config['steps']:
                    if not torch.isfinite(losses).all() or not torch.isfinite(parameter.coefficients).all():
                        raise RuntimeError('Nonfinite optimization')
                    history.append({'step': step+1, 'scores': scores.detach().cpu().tolist(),
                                    'loss': losses.detach().cpu().tolist(), 'path_loss': path_loss.detach().cpu().tolist(),
                                    'diversity': div.detach().cpu().tolist(), 'cutoff': cutoff, 'blur': sigma, 'lr': lr})
                if step+1 in config['save_steps']:
                    evaluation[str(step+1)] = evaluate(f'checkpoint_{step+1}')
                    write_json(output/'evaluation.json', evaluation)
                    atomic_torch(checkpoint, {'parameter': parameter.state_dict(), 'optimizer': optimizer.state_dict(),
                                 'active': active, 'first': first, 'step': step+1, 'history': history,
                                 'rngs': [rng.getstate() for rng in rngs]})
                if step == start or (step+1) % 128 == 0 or step+1 == config['steps']:
                    write_json(output/'progress.json', {'step': step+1, 'steps': config['steps'],
                               'elapsed_seconds': time.monotonic()-started, 'layer': layer, 'slot': slot, 'channels': channels})
            torch.cuda.synchronize()
            elapsed = time.monotonic()-started
            evaluation['final'] = evaluate('final')
            write_json(output/'evaluation.json', evaluation)
            write_json(output/'history.json', history)
            metadata = {'model': self.provenance, 'config': config, 'slot': slot, 'members': members,
                        'layer': layer, 'previous_layer': capture.previous, 'channels': channels,
                        'signed_mapping': [capture.mapping[c] for c in channels],
                        'activation': 'pre-ReLU' if negative else 'post-ReLU',
                        'direction': 'minimize' if negative else 'maximize',
                        'aggregation': 'normalized Gaussian, coordinates [-.5,.5]',
                        'source_sha256': self.source_hashes['full_feature_viz.py'],
                        'source_hashes': self.source_hashes, 'seed_noise_sha256': parameter.seed_hashes,
                        'versions': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'numpy': np.__version__},
                        'color': self.color_provenance, 'created_at': datetime.now(timezone.utc).isoformat(),
                        'path_definition': 'previous catalog layer + current post-ReLU Gaussian-pooled vectors, target channel zeroed; unit mean of top60 natural paths; first64 steps only',
                        'diversity_definition': 'previous-layer Gaussian-weighted Gram cosine within each four-image group only',
                        'anchor_protocol_sha256': anchors.get('_protocol_sha256') if anchors else None,
                        'scope': 'A parameterization extended to other catalog layers and five variants; not recovered author hyperparameters',
                        'batching': 'per-neuron independent Adam slices, augmentation streams initialized from the configured seed, per-neuron ReLU release state; no cross-neuron diversity'}
            write_json(output/'metadata.json', metadata)
            result = {'layer': layer, 'channels': channels, 'slot': slot, 'groups': groups, 'members': members,
                      'steps': config['steps'], 'executed_steps': config['steps']-start, 'elapsed_seconds': elapsed,
                      'image_steps_per_second': groups*members*(config['steps']-start)/max(elapsed, 1e-8),
                      'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
                      'peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2,
                      'first_positive_step': first.tolist()}
            write_json(output/'complete.json', result)
            print(json.dumps(result), flush=True)
            return result
        finally:
            self.redirect.enabled = None
            capture.close()
            del parameter, optimizer
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layer', choices=[x[0] for x in LAYERS], default='mixed4c')
    p.add_argument('--channels', nargs='+', type=int, default=[447])
    p.add_argument('--slot', choices=SLOTS, default='positive')
    p.add_argument('--steps', type=int, default=1024)
    p.add_argument('--anchors', type=Path)
    p.add_argument('--fitted-color', type=Path, default=DEFAULT_COLOR)
    a = p.parse_args()
    anchors = torch.load(a.anchors, map_location='cpu', weights_only=True) if a.anchors else None
    BatchRunner(a.fitted_color).run(a.output, a.layer, a.channels, a.slot, config_A(a.steps), anchors)


if __name__ == '__main__':
    main()
