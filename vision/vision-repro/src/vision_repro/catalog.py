"""Lightweight public model catalog: no CUDA or torch import required."""
LAYERS = [('conv2d0',64,112),('conv2d1',64,56),('conv2d2',192,56),
          ('mixed3a',256,28),('mixed3b',480,28),('mixed4a',508,14),
          ('mixed4b',512,14),('mixed4c',512,14),('mixed4d',528,14),
          ('mixed4e',832,14),('mixed5a',832,7),('mixed5b',1024,7)]
SLOTS = ('positive','path_positive','diversity','negative','path_negative')


def validate_target(layer, channels):
    counts = {name: count for name, count, _ in LAYERS}
    if layer not in counts or not channels:
        raise ValueError('Specify a known layer and at least one neuron')
    if len(set(channels)) != len(channels):
        raise ValueError('Duplicate neurons are not allowed in a production batch')
    if any(c < 0 or c >= counts[layer] for c in channels):
        raise ValueError(f'{layer}: neuron must be in [0, {counts[layer]-1}]')
