"""Capture selected MLP channels and index their native BF16 values."""
import contextlib
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .corpus import read, write


@contextlib.contextmanager
def channels(model, neurons):
    grouped = defaultdict(list)
    for column, unit in enumerate(neurons):
        grouped[unit['layer']].append((column, unit['neuron']))
    observed, handles = {}, []
    try:
        for layer, units in grouped.items():
            mlp = model.model.layers[layer].mlp
            indices = [unit for _, unit in units]

            def hook(name, indices=indices, layer=layer):
                def save(module, args, output=None):
                    import torch
                    tensor = args[0] if name == 'product' else output
                    selected = tensor.index_select(-1, torch.tensor(indices, device=tensor.device))
                    if selected.dtype != torch.bfloat16:
                        raise ValueError('Activation collection requires a BF16 model')
                    observed[layer, name] = selected.detach().cpu().contiguous().view(torch.uint16).numpy()
                return save

            handles.append(mlp.gate_proj.register_forward_hook(hook('gate')))
            handles.append(mlp.up_proj.register_forward_hook(hook('value')))
            handles.append(mlp.down_proj.register_forward_pre_hook(hook('product')))
        yield grouped, observed
    finally:
        for handle in handles:
            handle.remove()


def collect(root, model):
    import torch
    root = Path(root)
    manifest, neurons = read(root / 'dataset.json'), read(root / 'neurons.json')
    request = read(root / 'request.json')
    state_path = root / 'capture-state.json'
    state = read(state_path) if state_path.exists() else {'completed_shards': [], 'seconds': 0.}
    folder = root / 'part-0'
    folder.mkdir(exist_ok=True)
    device = model.model.embed_tokens.weight.device
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with channels(model, neurons) as (grouped, observed), torch.inference_mode():
        for shard in manifest['parts'][0]['shards']:
            if shard['id'] in state['completed_shards']:
                continue
            began = time.perf_counter()
            prefix = root / 'inputs' / f'block-{shard["id"]:06d}'
            ids = np.load(str(prefix) + '.ids.u32.npy', mmap_mode='r', allow_pickle=False)
            offsets = np.load(str(prefix) + '.offsets.u64.npy', allow_pickle=False)
            temporary = folder / (shard['path'] + '.tmp')
            output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint16,
                                               shape=(len(neurons), shard['token_count'], 3))
            try:
                start = 0
                while start < len(offsets) - 1:
                    stop = start + 1
                    longest = int(offsets[stop] - offsets[start])
                    while stop < len(offsets) - 1 and stop - start < request['batch_size']:
                        next_length = int(offsets[stop + 1] - offsets[stop])
                        next_longest = max(longest, next_length)
                        if next_longest * (stop - start + 1) > request['batch_tokens']:
                            break
                        longest = next_longest
                        stop += 1
                    lengths = np.diff(offsets[start:stop + 1]).astype(np.int64)
                    batch = torch.zeros((stop - start, longest), dtype=torch.long, device=device)
                    mask = torch.zeros_like(batch)
                    for row, length in enumerate(lengths):
                        a, b = map(int, offsets[start + row:start + row + 2])
                        batch[row, :length] = torch.as_tensor(np.array(ids[a:b], dtype=np.int64), device=device)
                        mask[row, :length] = 1
                    observed.clear()
                    model.model(input_ids=batch, attention_mask=mask, use_cache=False)
                    for layer, units in grouped.items():
                        packed = np.stack([observed[layer, key] for key in ('gate', 'value', 'product')], axis=-1)
                        columns = [column for column, _ in units]
                        for row, length in enumerate(lengths):
                            a, b = map(int, offsets[start + row:start + row + 2])
                            output[columns, a:b, :] = packed[row, :length].transpose(1, 0, 2)
                    start = stop
                output.flush()
            finally:
                output._mmap.close()
                ids._mmap.close()
            temporary.replace(folder / shard['path'])
            state['completed_shards'].append(shard['id'])
            state['seconds'] += time.perf_counter() - began
            write(state_path, state)
            write(root / 'status.json', {'stage': 'capture', 'completed_shards': len(state['completed_shards']),
                                        'shards': len(manifest['parts'][0]['shards']),
                                        'tokens': shard['token_start'] + shard['token_count']})


def logit_effects(root, model, tokenizer, count=12):
    import torch
    root = Path(root)
    neurons = read(root / 'neurons.json')
    pending = [unit for unit in neurons if not (root / 'effects' / (unit['id'] + '.json')).exists()]
    if not pending:
        return
    with torch.inference_mode():
        head = model.lm_head.weight
        vectors = torch.stack([model.model.layers[unit['layer']].mlp.down_proj.weight[:, unit['neuron']]
                               .to(device=head.device, dtype=torch.float32) for unit in pending], dim=1)
        scores = np.empty((head.shape[0], len(pending)), np.float32)
        for start in range(0, len(head), 8192):
            scores[start:start + 8192] = (head[start:start + 8192].float() @ vectors).cpu().numpy()
        norms = vectors.norm(dim=0).cpu().tolist()
    for column, unit in enumerate(pending):
        values = scores[:, column]

        def rows(high):
            order = np.argsort(-values if high else values, kind='stable')[:count]
            return [{'token_id': int(i), 'token': tokenizer.decode([int(i)], skip_special_tokens=False)
                     or f'⟦token {i}⟧', 'weight': float(values[i])} for i in order]

        write(root / 'effects' / (unit['id'] + '.json'), {'logit_effect': {
            'promote': rows(True), 'suppress': rows(False), 'norm': norms[column],
            'method': 'lm_head.weight @ down_proj.weight[:, neuron]'}})


def index(root):
    from language.analysis.build_report import best, floats, statistics
    root = Path(root)
    manifest, neurons = read(root / 'dataset.json'), read(root / 'neurons.json')
    if manifest['complete']:
        return
    state = read(root / 'capture-state.json')
    shards = manifest['parts'][0]['shards']
    if set(state['completed_shards']) != {shard['id'] for shard in shards}:
        raise ValueError('Activation collection is incomplete')
    size, k = len(neurons), manifest['indices']['k']
    histogram = np.zeros((size, 3, 65536), np.uint64)
    high_ids = np.empty((size, 3, 0), np.int64)
    low_ids = high_ids.copy()
    high_values = np.empty((size, 3, 0), np.float32)
    low_values = high_values.copy()
    for shard in shards:
        bits = np.load(root / 'part-0' / shard['path'], mmap_mode='r', allow_pickle=False)
        try:
            if bits.dtype != np.uint16 or bits.shape != (size, shard['token_count'], 3):
                raise ValueError('Activation shard dimensions differ from the manifest')
            local_ids, local_values = {}, {}
            for high in (True, False):
                local_ids[high] = np.empty((size, 3, min(k, shard['token_count'])), np.int64)
                local_values[high] = np.empty(local_ids[high].shape, np.float32)
            for column in range(size):
                if np.any((bits[column] & 0x7f80) == 0x7f80):
                    raise ValueError('Non-finite native activation')
                for channel in range(3):
                    raw = bits[column, :, channel]
                    histogram[column, channel] += np.bincount(raw, minlength=65536).astype(np.uint64)
                    values = floats(raw)
                    for high in (True, False):
                        selected = best(values, k, high)
                        local_ids[high][column, channel] = selected + shard['token_start']
                        local_values[high][column, channel] = values[selected]
            def merge(ids, values, high):
                ids = np.concatenate([ids, local_ids[high]], axis=-1)
                values = np.concatenate([values, local_values[high]], axis=-1)
                order = np.lexsort((ids, -values if high else values), axis=-1)[..., :k]
                return np.take_along_axis(ids, order, -1), np.take_along_axis(values, order, -1)
            high_ids, high_values = merge(high_ids, high_values, True)
            low_ids, low_values = merge(low_ids, low_values, False)
        finally:
            bits._mmap.close()
    if not np.all(histogram.sum(axis=2) == manifest['tokens']):
        raise ValueError('Activation counts differ from the token budget')
    np.save(root / 'part-0/histogram.npy', histogram)
    (root / 'indices').mkdir(exist_ok=True)
    np.savez(root / 'indices/extremes.npz', high_token_indices=high_ids, high_values=high_values,
             low_token_indices=low_ids, low_values=low_values)
    rows = []
    for column, unit in enumerate(neurons):
        for channel, name in enumerate(manifest['channels']):
            values = statistics(histogram[column, channel])
            rows.append({'neuron_id': unit['id'], 'channel': name,
                         **{key: values[key] for key in ('n', 'mean', 'std', 'min', 'max', 'positive', 'negative', 'zero')}})
    pq.write_table(pa.Table.from_pylist(rows), root / 'indices/statistics.parquet')
    write(root / 'verification.json', {'complete': True, 'tokens': manifest['tokens'],
                                     'documents': manifest['documents'], 'neurons': size, 'shards': len(shards)})
    manifest['complete'] = True
    write(root / 'dataset.json', manifest)
