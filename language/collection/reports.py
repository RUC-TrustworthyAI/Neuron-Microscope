"""Build investigation-ready reports with the existing saved-data renderer."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import shutil
import sys

import numpy as np
from tokenizers import Tokenizer

from .corpus import read, write


def _initialize(root):
    root = Path(root)
    language = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(language / 'analysis'))
    sys.path.insert(0, str(language / 'tools'))
    import build_dataset_reports as builder
    builder.initialize(root, root / 'inputs', Path(read(root / 'request.json')['model']) / 'tokenizer.json',
                       root / 'reports/.building', read(root / 'request.json')['seed'], True, root / 'token-cache')


def _build(unit):
    import build_dataset_reports as builder
    from report import merge
    root = builder.API.root
    temporary = builder.OUTPUT / unit['id']
    if temporary.exists():
        shutil.rmtree(temporary)
    entry = builder.build_one(unit)
    effects = read(root / 'effects' / (unit['id'] + '.json'))
    write(temporary / 'effects.json', effects)
    page = temporary / 'index.html'
    source, _ = merge(page.read_text(), {'logit_effect': effects['logit_effect']})
    page.write_text(source)
    temporary.rename(root / 'reports' / unit['id'])
    return entry


def build(root, workers=2):
    root = Path(root)
    report_root = root / 'reports'
    if (report_root / 'complete.json').exists():
        return
    neurons, manifest = read(root / 'neurons.json'), read(root / 'dataset.json')
    cache = root / 'token-cache'
    cache.mkdir(exist_ok=True)
    tokenizer = Tokenizer.from_file(str(Path(read(root / 'request.json')['model']) / 'tokenizer.json'))
    counts = np.zeros(tokenizer.get_vocab_size(with_added_tokens=True), np.int64)
    ids = np.load(root / 'token_ids.npy', mmap_mode='r', allow_pickle=False)
    try:
        for start in range(0, len(ids), 1_000_000):
            counts += np.bincount(ids[start:start + 1_000_000], minlength=len(counts))
    finally:
        ids._mmap.close()
    np.save(cache / 'token-counts.i64.npy', counts)
    write(cache / 'token-labels.json', [tokenizer.decode([i], skip_special_tokens=False) for i in range(len(counts))])
    (report_root / '.building/jobs').mkdir(parents=True, exist_ok=True)
    entries, pending = [], []
    for unit in neurons:
        folder = report_root / unit['id']
        if folder.exists():
            entries.append(read(folder / 'complete.json'))
        else:
            pending.append(unit)
    write(root / 'status.json', {'stage': 'reports', 'neurons_complete': len(entries), 'neurons': len(neurons)})
    if workers == 1:
        _initialize(root)
        for unit in pending:
            entries.append(_build(unit))
            write(root / 'status.json', {'stage': 'reports', 'neurons_complete': len(entries), 'neurons': len(neurons)})
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                 initializer=_initialize, initargs=(root,)) as pool:
            futures = [pool.submit(_build, unit) for unit in pending]
            for future in as_completed(futures):
                entries.append(future.result())
                write(root / 'status.json', {'stage': 'reports', 'neurons_complete': len(entries),
                                            'neurons': len(neurons)})
    # The single-worker path and the parent process use the same catalog exporter.
    language = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(language / 'analysis'))
    import build_dataset_reports as builder
    entries.sort(key=lambda row: tuple(int(x) for x in row['id'][1:].split('_N')))
    builder.publish_catalog(report_root, entries, True)
    write(report_root / 'complete.json', {'complete': True, 'neurons': len(entries),
                                        'tokens_per_neuron': manifest['tokens'], 'documents': manifest['documents']})

