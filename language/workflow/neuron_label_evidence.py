"""Sample related neurons' own native responses to saved corpus documents."""
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from hashlib import sha256
from pathlib import Path

import numpy as np


TOKEN_BUDGET = 32_768
QUOTAS = {'discovery': (5, 5, 3), 'holdout': (3, 3, 2)}
KINDS = ('positive', 'negative', 'control')


def needs_runtime(config, candidates):
    from dataset_reader import open_dataset

    dataset = open_dataset(config['activations'])
    try:
        return any(unit not in dataset.locations for unit in candidates)
    finally:
        dataset.close()


def documents(dataset, budget, seed):
    """Choose full saved prefixes, keeping duplicates and source windows together."""
    selected, seen, consumed = [], set(), 0
    for index in np.random.default_rng(seed).permutation(len(dataset.documents)):
        row = dataset.documents.slice(int(index), 1).to_pylist()[0]
        start, count = int(row['token_start']), int(row['token_count'])
        if count == 0 or count > budget:
            continue
        if consumed + count > budget:
            break
        ids = dataset.token_ids[start:start + count]
        fingerprint = sha256(ids.tobytes()).digest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        group = (row.get('source_file'), row.get('input_row'))
        if None in group:
            group = fingerprint
        selected.append((int(row['document_id']), start, list(map(int, ids)), group))
        consumed += count
        if consumed == budget:
            break
    groups = list(dict.fromkeys(row[3] for row in selected))
    held_out = set(groups[:max(1, len(groups) // 5)]) if len(groups) > 1 else set()
    return [(doc, start, ids, 'holdout' if group in held_out else 'discovery')
            for doc, start, ids, group in selected]


@contextmanager
def products(runtime, units):
    """Read all selected coordinates in each forward without the vocabulary projection."""
    torch = runtime.torch
    layers, captured, hooks = defaultdict(list), {}, []
    for unit in units:
        layer, neuron = map(int, unit[1:].split('_N'))
        layers[layer].append((unit, neuron))

    def capture(rows, device):
        indices = torch.tensor([neuron for _, neuron in rows], device=device)

        def hook(_, args):
            values = args[0].index_select(-1, indices).detach().float().cpu().numpy()
            captured.update({unit: values[:, :, i] for i, (unit, _) in enumerate(rows)})

        return hook

    def forward(inputs):
        width = max(map(len, inputs))
        device = runtime.model.model.embed_tokens.weight.device
        ids = torch.tensor([seq + [0] * (width - len(seq)) for seq in inputs], device=device)
        mask = torch.tensor([[1] * len(seq) + [0] * (width - len(seq)) for seq in inputs], device=device)
        captured.clear()
        with torch.inference_mode():
            runtime.model.model(input_ids=ids, attention_mask=mask, use_cache=False)
        runtime.forwards += 1
        return [{unit: values[i, :len(seq)] for unit, values in captured.items()}
                for i, seq in enumerate(inputs)]

    try:
        for layer, rows in layers.items():
            down = runtime.model.model.layers[layer].mlp.down_proj
            hooks.append(down.register_forward_pre_hook(capture(rows, down.weight.device)))
        yield forward
    finally:
        for hook in hooks:
            hook.remove()


def batches(rows, batch_size, batch_tokens):
    batch = []
    for row in sorted(rows, key=lambda row: len(row[2])):
        width = len(row[2])
        if batch and (len(batch) >= batch_size or width * (len(batch) + 1) > batch_tokens):
            yield batch
            batch = []
        batch.append(row)
    if batch:
        yield batch


def add_peaks(pool, document, values, source):
    doc, start, ids, split = document
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) != len(ids):
        raise ValueError('Label evidence must have a finite native activation for each input token')
    hi, lo = int(np.argmax(values)), int(np.argmin(values))
    controls = finite[(finite != hi) & (finite != lo)]
    low = int(controls[np.argmin(np.abs(values[controls]))]) if len(controls) else None
    for kind, index in zip(KINDS, (hi, lo, low)):
        if index is None:
            continue
        value = float(values[index])
        if (kind == 'positive' and value <= 0) or (kind == 'negative' and value >= 0):
            continue
        score = -abs(value) if kind == 'control' else abs(value)
        pool[split, kind].append((score, doc, start, index, value, ids, source))


def evidence_context(pieces, pending, position):
    start, stop = max(0, position - 96), min(len(pieces), position + 9)
    prefix = ''.join(pieces[start:position])
    token = pieces[position]
    suffix = ''.join(pieces[position + 1:stop])
    if pending[stop - 1] and stop < len(pieces):
        suffix += next((piece[0] for piece in pieces[stop:] if piece), '')
    if pending[position] and suffix:
        # Highlight the whole character containing the anchor's trailing bytes.
        token, suffix = token + suffix[0], suffix[1:]
    return prefix, token, suffix


def samples(unit, pool, decoded):
    rows = []
    for split, quotas in QUOTAS.items():
        for kind, quota in zip(KINDS, quotas):
            best = sorted(pool[split, kind], key=lambda entry: (-entry[0], entry[1]))[:quota]
            for _, doc, start, position, value, ids, source in best:
                prefix, token, suffix = evidence_context(*decoded[doc], position)
                rows.append({'id': f'{unit}:{start + position}', 'neuron_id': unit, 'document': doc,
                             'position': position, 'dataset_token_index': start + position,
                             'token_id': ids[position], 'token': token, 'prefix': prefix, 'suffix': suffix,
                             'context': prefix + token + suffix, 'activation': value, 'kind': kind,
                             'split': split, 'source': source})
    return rows


def collect(config, candidates, runtime=None, *, token_budget=TOKEN_BUDGET):
    """Replay a bounded corpus slice once for units not already in the native archive."""
    from dataset_reader import open_dataset
    from tokenizers import Tokenizer
    from language.analysis.build_report import token_pieces

    candidates = list(candidates)
    if token_budget <= 0:
        raise ValueError('Neuron label evidence requires a positive token budget')
    settings = config.get('settings', {})
    if runtime is not None and hasattr(runtime, 'call'):
        capture_config = {key: config[key] for key in
                          ('model', 'activations', 'tokenizer', 'batch_size', 'batch_tokens') if key in config}
        capture_config['settings'] = {'seed': settings.get('seed', 42)}
        return runtime.call('neuron-label-evidence', config=capture_config,
                            candidates=candidates, token_budget=token_budget)

    dataset = open_dataset(config['activations'])
    try:
        saved = [unit for unit in candidates if unit in dataset.locations]
        missing = [unit for unit in candidates if unit not in dataset.locations]
        if missing and runtime is None:
            raise ValueError('Neuron label evidence requires a model for units absent from the archive')
        seed = settings.get('seed', 42)
        selected = documents(dataset, token_budget, seed)
        capture = dataset.manifest.get('capture', {})
        batch_size, batch_tokens = capture.get('batch_size', 1), capture.get('batch_tokens', 2048)
        batch_size = min(batch_size, config.get('batch_size', batch_size))
        batch_tokens = min(batch_tokens, config.get('batch_tokens', batch_tokens))
        pools = {unit: defaultdict(list) for unit in candidates}
        scanner = products(runtime, missing) if missing else nullcontext(None)
        with scanner as forward:
            for batch in batches(selected, batch_size, batch_tokens):
                observed = forward([row[2] for row in batch]) if forward else [{} for _ in batch]
                for row, live in zip(batch, observed):
                    for unit in saved:
                        record = dataset.document_activations(row[0], unit)
                        if not np.array_equal(record['token_ids'], row[2]):
                            raise ValueError('Archive label evidence does not match its saved token IDs')
                        channels = record['channels']
                        channel = channels.index('product' if 'product' in channels else 'native_product')
                        add_peaks(pools[unit], row, record['activations'][:, channel], 'archive')
                    for unit in missing:
                        add_peaks(pools[unit], row, live[unit], 'measured')
        model_path = Path(config['model']).resolve()
        tokenizer = Tokenizer.from_file(str(config.get('tokenizer') or model_path / 'tokenizer.json'))
        decoded = {}
        for doc, _, ids, _ in selected:
            pieces, _, pending = token_pieces(tokenizer, ids)
            decoded[doc] = pieces, pending
        measurement = None
        if missing:
            measurement = {'model': getattr(runtime, 'identity', {'path': str(model_path)}),
                           'dtype': runtime.settings.get('dtype'), 'attention': runtime.settings.get('attention'),
                           'batch_size': batch_size, 'batch_tokens': batch_tokens}
        scope = {'archive': str(Path(config['activations']).resolve()),
                 'archive_model': dataset.manifest.get('model'),
                 'model_snapshot': dataset.manifest.get('model_snapshot'),
                 'capture': capture, 'measurement': measurement, 'seed': seed, 'token_budget': token_budget,
                 'corpus_tokens': dataset.manifest['tokens'], 'archive_units': saved, 'measured_units': missing,
                 'document_ids': {split: sorted(doc for doc, _, _, s in selected if s == split) for split in QUOTAS},
                 'token_counts': {split: sum(len(ids) for _, _, ids, s in selected if s == split) for split in QUOTAS}}
        return {'scope': scope, 'samples': {unit: samples(unit, pool, decoded) for unit, pool in pools.items()}}
    finally:
        dataset.close()
