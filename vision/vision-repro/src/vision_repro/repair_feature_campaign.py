"""Retry audited zero-activation variants with new seeds and an unchanged recipe.

Each seed has an independent resumable queue. Only numerically valid complete
variants enter accepted.json; source results and previous attempts are retained.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from PIL import Image


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def key(layer, channel, slot):
    return f'{layer}:{channel}:{slot}'


def rows(root):
    with sqlite3.connect(f'file:{root}/queue.sqlite?mode=ro', uri=True) as con:
        con.row_factory = sqlite3.Row
        return [dict(row) for row in con.execute('SELECT * FROM jobs ORDER BY rowid')]


def audited_targets(base, audit):
    jobs = rows(base)
    if any(j['state'] in ('pending', 'running') for j in jobs):
        raise ValueError('Baseline campaign must have finished before quality repair')
    index = {}
    for job in jobs:
        if job['state'] == 'done':
            for i, channel in enumerate(json.loads(job['channels'])):
                index[key(job['layer'], channel, job['slot'])] = (job['id'], i)
    targets, seen, cache = [], set(), {}
    for item in read(audit)['flagged_variants']:
        layer, channel, slot = item['layer'], item['neuron'], item['slot']
        target = key(layer, channel, slot)
        if slot not in ('positive', 'diversity') or target in seen:
            raise ValueError(f'Unsupported or duplicate repair target: {target}')
        seen.add(target)
        job, i = index[target]
        if job not in cache:
            cache[job] = read(base/'batches'/job/'evaluation.json')
        scores = cache[job]['final'][i]['score']
        if scores != item['scores'] or not any(v <= 0 for v in scores):
            raise ValueError(f'Audit differs from completed baseline: {target}')
        targets.append({'layer': layer, 'channel': channel, 'slot': slot,
                        'baseline_job': job, 'baseline_index': i, 'baseline_scores': scores})
    return sorted(targets, key=lambda t: (t['slot'] != 'positive', t['layer'], t['channel']))


def prepare_round(root, base, targets, seed, batch_images):
    from .full_feature_campaign import database
    if seed <= 0 or batch_images <= 0 or batch_images % 4:
        raise ValueError('Use a positive seed and a positive multiple of four images')
    root.mkdir(parents=True, exist_ok=True)
    identity = {'targets': targets, 'seed': seed, 'batch_images': batch_images,
                'baseline': str(base), 'baseline_protocol_sha256': digest(base/'protocol.json')}
    if (root/'selection.json').exists():
        if read(root/'selection.json') != identity:
            raise ValueError('Refusing to change an existing repair round')
    else:
        save(root/'selection.json', identity)
    anchors = root/'anchors'
    if not anchors.exists():
        anchors.symlink_to((base/'anchors').resolve(), target_is_directory=True)
    protocol = read(base/'protocol.json')
    protocol['config']['seed'] = seed
    protocol.update(neurons=len({(t['layer'], t['channel']) for t in targets}),
                    requested_neuron_variants=len(targets), batch_images=batch_images,
                    single_image_neurons_per_batch=batch_images,
                    diversity_neurons_per_batch=batch_images//4,
                    physical_images=sum(4 if t['slot'] == 'diversity' else 1 for t in targets))
    protocol['quality_repair'] = identity
    if (root/'protocol.json').exists():
        if read(root/'protocol.json') != protocol:
            raise ValueError('Repair protocol changed')
    else:
        save(root/'protocol.json', protocol)
    con = database(root)
    # A transaction makes queue creation all-or-nothing. Never repopulate a live queue.
    if not con.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]:
        groups = defaultdict(list)
        for t in targets:
            groups[t['layer'], t['slot']].append(t['channel'])
        for (layer, slot), channels in groups.items():
            width = batch_images // (4 if slot == 'diversity' else 1)
            for start in range(0, len(channels), width):
                selected = channels[start:start+width]
                jid = f'{layer}_{slot}_{start:04d}_{len(selected):02d}'
                con.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)',
                            (jid, layer, slot, json.dumps(selected), 'pending', None, None, time.time()))
        con.commit()
    con.close()


def validate_batch(root, row):
    folder = root/'batches'/row['id']
    complete, meta, evaluation = [read(folder/name) for name in
                                  ('complete.json', 'metadata.json', 'evaluation.json')]
    cfg = read(root/'protocol.json')['config']
    channels = json.loads(row['channels'])
    slot, layer = row['slot'], row['layer']
    members = 4 if slot == 'diversity' else 1
    expected_cfg = {k: v for k, v in cfg.items() if k not in ('channel', 'slot')}
    expected_cfg.update(layer=layer, slot=slot)
    if not (complete['channels'] == meta['channels'] == channels and
            complete['slot'] == meta['slot'] == slot and
            complete['layer'] == meta['layer'] == layer and
            complete['members'] == meta['members'] == members and
            complete['steps'] == cfg['steps'] and meta['config'] == expected_cfg and
            meta['activation'] == 'post-ReLU' and len(evaluation['final']) == len(channels)):
        raise ValueError(f'Mismatched batch provenance: {folder}')
    if not (folder/'history.json').is_file() or not (folder/'resume.pt').is_file():
        raise ValueError(f'Missing history/checkpoint: {folder}')
    accepted, rejected = {}, {}
    for i, channel in enumerate(channels):
        target = key(layer, channel, slot)
        scores = evaluation['final'][i]['score']
        if len(scores) != members or not all(math.isfinite(v) for v in scores):
            raise ValueError(f'Invalid scores: {target}')
        if any(v <= 0 for v in scores):
            rejected[target] = scores
            continue
        item = folder/'items'/str(i)
        paths = [item/'final.png'] + ([item/f'final_member{j}.png' for j in range(4)] if members == 4 else [])
        images = []
        for j, path in enumerate(paths):
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                expected = cfg['size'] * (2 if members == 4 and j == 0 else 1)
                if im.size != (expected, expected):
                    raise ValueError(f'Wrong image dimensions: {path}')
                images.append(im.convert('RGB'))
        if members == 4:
            size = cfg['size']
            for j, member in enumerate(images[1:]):
                box = (j%2*size, j//2*size, (j%2+1)*size, (j//2+1)*size)
                if images[0].crop(box).tobytes() != member.tobytes():
                    raise ValueError(f'Diversity preview/member mismatch: {target}')
        accepted[target] = {'layer': layer, 'channel': channel, 'slot': slot,
                            'seed': cfg['seed'], 'scores': scores, 'batch': str(folder), 'index': i,
                            'metadata_sha256': digest(folder/'metadata.json'),
                            'evaluation_sha256': digest(folder/'evaluation.json'),
                            'images_sha256': {p.name: digest(p) for p in paths}}
    return accepted, rejected


def run(base, audit, output, seeds, workers, batch_images):
    output.mkdir(parents=True, exist_ok=True)
    with (output/'repair.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        targets = audited_targets(base, audit)
        plan = {'baseline': str(base), 'audit_sha256': digest(audit), 'targets': targets,
                'seeds': seeds, 'workers': workers, 'batch_images': batch_images,
                'coordinator_sha256': digest(__file__),
                'recipe': 'Original A/1024 unchanged except seed; rerun whole diversity group; preserve every attempt',
                'acceptance': 'Finite positive final post-ReLU score for every member and valid image/provenance files'}
        if (output/'plan.json').exists() and read(output/'plan.json') != plan:
            raise ValueError('Refusing to change a recorded repair plan')
        save(output/'plan.json', plan)
        accepted = {}
        state = {'requested': len(targets), 'accepted': 0, 'remaining': len(targets)}
        for seed in seeds:
            pending = [t for t in targets if key(t['layer'], t['channel'], t['slot']) not in accepted]
            if not pending:
                break
            root = output/f'seed_{seed}'
            prepare_round(root, base, pending, seed, batch_images)
            seen, rejected = set(), {}
            child = None
            try:
                with (root/'supervisor.log').open('a') as log:
                    child = subprocess.Popen([sys.executable, '-m', 'vision_repro.full_feature_supervisor',
                                              '--root', str(root), '--workers', str(workers)],
                                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    while True:
                        jobs = rows(root)
                        for row in jobs:
                            if row['state'] == 'done' and row['id'] not in seen:
                                good, bad = validate_batch(root, row)
                                accepted.update(good)
                                rejected.update(bad)
                                seen.add(row['id'])
                        save(output/'accepted.json', accepted)
                        state = {'updated_at': datetime.now(timezone.utc).isoformat(),
                                 'seed': seed, 'requested': len(targets), 'accepted': len(accepted),
                                 'remaining': len(targets)-len(accepted), 'round_rejected': rejected,
                                 'round_root': str(root), 'status': 'running'}
                        save(output/'progress.json', state)
                        code = child.poll()
                        if code is not None:
                            # Completion is written after the final worker updates its row.
                            if any(j['state'] == 'done' and j['id'] not in seen for j in rows(root)):
                                continue
                            if code or any(j['state'] in ('failed', 'running', 'pending') for j in jobs):
                                raise RuntimeError(f'Repair execution failed; inspect {root}')
                            break
                        time.sleep(20)
            finally:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
        state.update(status='complete' if len(accepted) == len(targets) else 'needs_review',
                     unresolved=[t for t in targets if key(t['layer'], t['channel'], t['slot']) not in accepted])
        save(output/'progress.json', state)
        save(output/'repair_finished.json', state)
        print(json.dumps(state), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-images', type=int, default=32)
    a = p.parse_args()
    if a.workers < 1 or len(set(a.seeds)) != len(a.seeds) or any(s <= 0 for s in a.seeds):
        p.error('Use positive workers and unique positive seeds')
    try:
        run(a.base.resolve(), a.audit.resolve(), a.output.resolve(), a.seeds, a.workers, a.batch_images)
    except BlockingIOError:
        raise  # Another coordinator owns this output; do not alter its records.
    except Exception as exc:
        if a.output.exists():
            save(a.output/'error.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
