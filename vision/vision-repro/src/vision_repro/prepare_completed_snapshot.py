"""Export finished jobs, optionally retaining zero-activation images."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from PIL import Image
from .site_export import public_metadata
from .repair_feature_campaign import validate_batch, rows as repair_rows

SLOTS = ['positive', 'path_positive', 'diversity', 'negative', 'path_negative']
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()

def activation_valid(slot, info):
    scores = info.get('final', {}).get('score', [])
    members = 4 if slot == 'diversity' else 1
    return len(scores) == members and all(
        math.isfinite(v) and (v > 0 if slot in SLOTS[:3] else v < 0)
        for v in scores)

def complete_keys(tasks, base_nodes=None):
    """Require all five valid modes and every original Diversity member."""
    present = {k: set(n['ours']) for k,n in (base_nodes or {}).items()}
    for key, slot, folder, i, members, *_ in tasks:
        files = [folder / f'items/{i}/final.png']
        if members == 4:
            files += [folder / f'items/{i}/final_member{j}.png' for j in range(4)]
        if all(p.is_file() for p in files):
            present.setdefault(key, set()).add(slot)
    return {key for key, slots in present.items() if slots == set(SLOTS)}

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--complete-neurons-only', action='store_true')
    p.add_argument('--include-zero-diversity', action='store_true',
                   help='Display all four Diversity members even when some scores are zero; retain actual scores.')
    p.add_argument('--include-zero-activations', action='store_true',
                   help='Retain zero-activation images in every mode, with their original scores and quality flags.')
    p.add_argument('--base-snapshot', type=Path, help='Preserve previously published nodes that are not eligible for an update.')
    p.add_argument('--repair-root', type=Path, help='Use an atomic snapshot of validated seed-retry results.')
    a = p.parse_args(argv)
    for name in ('root', 'output', 'base_snapshot', 'repair_root'):
        value = getattr(a, name)
        if value is not None:
            setattr(a, name, value.resolve())
    a.output.mkdir(parents=True, exist_ok=False)
    release = a.output.name
    base_nodes, base_records, base_manifest = {}, {}, None
    if a.base_snapshot:
        base_manifest = json.loads((a.base_snapshot / 'upload-manifest.json').read_bytes())
        base_release = base_manifest['summary']['release']
        base_records = {r['key']: r for r in base_manifest['objects']}
        catalog = json.loads((a.base_snapshot / f'releases/{base_release}/catalog.json').read_bytes())
        for key in catalog['nodes']:
            layer, channel = key.split(':')
            rel = f'releases/{base_release}/nodes/{layer}/{channel}.json'
            raw = (a.base_snapshot / rel).read_bytes()
            assert sha(raw) == base_records[rel]['sha256'] and len(raw) == base_records[rel]['bytes']
            base_nodes[key] = json.loads(raw)
    def reuse_image(info):
        rel = f'images/{info["image_sha256"]}.webp'
        record = base_records[rel]
        source = a.base_snapshot / rel
        raw = source.read_bytes()
        assert sha(raw) == record['sha256'] and len(raw) == record['bytes']
        target = a.output / rel
        target.parent.mkdir(exist_ok=True)
        try:
            os.link(source, target)
        except FileExistsError:
            pass
        return record
    now = datetime.now(timezone.utc).isoformat()
    protocol_raw = (a.root / 'protocol.json').read_bytes()
    protocol = json.loads(protocol_raw)
    for name, digest in protocol['source_sha256'].items():
        assert sha((a.root.parents[1] / 'src/vision_repro' / name).read_bytes()) == digest
    db = sqlite3.connect(f'file:{a.root}/queue.sqlite?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute("SELECT * FROM jobs WHERE state='done' ORDER BY rowid")]
    db.close()
    (a.output / 'queue-snapshot.json').write_bytes(compact({'snapshotAt': now, 'jobs': rows}))
    nodes, tasks, flags, files = {}, [], [], []
    for row in rows:
        folder = a.root / 'batches' / row['id']
        complete = json.loads((folder / 'complete.json').read_bytes())
        meta = json.loads((folder / 'metadata.json').read_bytes())
        evaluation = json.loads((folder / 'evaluation.json').read_bytes())
        channels = json.loads(row['channels'])
        slot, layer = row['slot'], row['layer']
        members = 4 if slot == 'diversity' else 1
        assert (folder / 'history.json').is_file()
        assert complete['channels'] == meta['channels'] == channels
        assert complete['slot'] == slot and complete['members'] == members
        assert complete['steps'] == meta['config']['steps'] == 1024
        assert meta['config']['seed'] == 0 and meta['layer'] == layer
        for i, channel in enumerate(channels):
            key = f'{layer}:{channel}'
            node = nodes.setdefault(key, {'layer': layer, 'channel': channel, 'model': public_metadata(meta['model']),
                'config': public_metadata(meta['config']), 'ours': {}, 'variantInfo': {}, 'pending': {}})
            scores = evaluation['final'][i]['score']
            assert len(scores) == members and all(math.isfinite(v) for v in scores)
            positive = slot in SLOTS[:3]
            wrong = [j for j,v in enumerate(scores) if (v <= 0 if positive else v >= 0)]
            if wrong:
                node['pending'][slot] = 'quality_review'
                flags.append({'neuron': key, 'slot': slot, 'scores': scores, 'members': wrong, 'job': row['id']})
                zero_allowed = a.include_zero_activations or (a.include_zero_diversity and slot == 'diversity')
                if not (zero_allowed and all(v >= 0 if positive else v <= 0 for v in scores)):
                    continue
            tasks.append((key, slot, folder, i, members, meta, evaluation))
    repair_count = 0
    if a.repair_root:
        raw = (a.repair_root/'accepted.json').read_bytes()
        (a.output/'repair-accepted-snapshot.json').write_bytes(raw)
        accepted = json.loads(raw)
        cache = {}
        replacements = {}
        for target, record in accepted.items():
            folder = Path(record['batch']).resolve()
            if not folder.is_relative_to(a.repair_root.resolve()):
                raise ValueError('Repair batch escapes its declared root')
            if folder not in cache:
                seed_root = folder.parent.parent
                row = next(r for r in repair_rows(seed_root) if r['id'] == folder.name)
                if row['state'] != 'done':
                    raise ValueError('Repair batch is not done')
                cache[folder] = validate_batch(seed_root, row)[0]
            if cache[folder].get(target) != record:
                raise ValueError('Repair receipt differs from verified artifacts')
            meta = json.loads((folder/'metadata.json').read_bytes())
            evaluation = json.loads((folder/'evaluation.json').read_bytes())
            key, slot, i = f"{record['layer']}:{record['channel']}", record['slot'], record['index']
            if key not in nodes or not any(f['neuron']==key and f['slot']==slot for f in flags):
                raise ValueError('Repair target was not a flagged baseline variant')
            replacements[key,slot] = (key,slot,folder,i,meta['members'],meta,evaluation)
            nodes[key]['pending'].pop(slot, None)
        tasks = [t for t in tasks if (t[0],t[1]) not in replacements] + list(replacements.values())
        flags = [f for f in flags if (f['neuron'],f['slot']) not in replacements]
        repair_count = len(replacements)
    discovered = len(nodes)
    if a.complete_neurons_only:
        eligible = complete_keys(tasks, base_nodes)
        tasks = [task for task in tasks if task[0] in eligible]
        nodes = {key: node for key, node in nodes.items() if key in eligible}
    updated_keys = set(nodes)
    # A continued campaign may contain only the modes missing from the published
    # baseline. Combine their coverage before selecting complete neurons.
    for key, node in nodes.items():
        if key in base_nodes:
            prior = copy.deepcopy(base_nodes[key])
            node['ours'] = prior['ours']
            node['variantInfo'] = prior['variantInfo']
            node['pending'] = prior['pending']
            files.extend(reuse_image(info) for info in prior['variantInfo'].values())
    def encode(task):
        key, slot, folder, i, members, meta, evaluation = task
        raw = (folder / f'items/{i}/final.png').read_bytes()
        previous = base_nodes.get(key, {}).get('variantInfo', {}).get(slot)
        # Reuse an already verified lossless export only if both pixels and
        # experiment provenance agree. Never substitute a different run.
        if previous and previous['source_png_sha256'] == sha(raw) and all([
            previous['config'] == public_metadata(meta['config']), previous['source_hashes'] == meta['source_hashes'],
            previous['versions'] == public_metadata(meta['versions']), previous['target'] == meta['signed_mapping'][i],
            previous['activation'] == meta['activation'], previous['aggregation'] == meta['aggregation'],
            previous['direction'] == meta['direction'], previous['batchSize'] == members,
            previous['initial'] == evaluation['initial'][i], previous['final'] == evaluation['final'][i],
        ]):
            if members == 4:
                with Image.open(io.BytesIO(raw)) as original:
                    rgb = original.convert('RGB')
                    assert rgb.size == (448, 448)
                    for j in range(4):
                        with Image.open(folder / f'items/{i}/final_member{j}.png') as member:
                            box = (j%2*224, j//2*224, (j%2+1)*224, (j//2+1)*224)
                            assert member.size == (224, 224) and rgb.crop(box).tobytes() == member.convert('RGB').tobytes()
            return key, slot, previous, reuse_image(previous)
        with Image.open(io.BytesIO(raw)) as original:
            rgb = original.convert('RGB')
            expected = 448 if members == 4 else 224
            assert rgb.size == (expected, expected), (key, slot)
            if members == 4:
                for j in range(4):
                    with Image.open(folder / f'items/{i}/final_member{j}.png') as member:
                        box = (j%2*224, j//2*224, (j%2+1)*224, (j//2+1)*224)
                        assert rgb.crop(box).tobytes() == member.convert('RGB').tobytes()
            output = io.BytesIO()
            rgb.save(output, format='WEBP', lossless=True, method=2)
            encoded = output.getvalue()
            with Image.open(io.BytesIO(encoded)) as verify:
                assert verify.convert('RGB').tobytes() == rgb.tobytes()
            pixels = sha(rgb.tobytes())
        image_sha = sha(encoded)
        rel = f'images/{image_sha}.webp'
        file = a.output / rel
        file.parent.mkdir(exist_ok=True)
        file.write_bytes(encoded)
        run_id = str(folder.relative_to(a.repair_root))+f'/items/{i}' if a.repair_root and folder.is_relative_to(a.repair_root) else f'{a.root.name}/batches/{folder.name}/items/{i}'
        if a.repair_root and folder.is_relative_to(a.repair_root):
            run_id = a.repair_root.name + '/' + run_id
        info = {'batchSize': members, 'runId': run_id,
            'activation': meta['activation'], 'aggregation': meta['aggregation'],
            'target': meta['signed_mapping'][i], 'direction': meta['direction'], 'config': public_metadata(meta['config']),
            'scores': {'1024': evaluation['final'][i]['score']}, 'initial': evaluation['initial'][i],
            'final': evaluation['final'][i], 'setting': 'A', 'image_sha256': image_sha,
            'source_png_sha256': sha(raw), 'pixel_sha256': pixels, 'image_size': [expected, expected],
            'source_hashes': meta['source_hashes'], 'versions': public_metadata(meta['versions'])}
        return key, slot, info, {'key': rel, 'sha256': image_sha, 'bytes': len(encoded)}
    started = time.monotonic()
    print(json.dumps({'snapshotAt': now, 'release': release, 'validVariants': len(tasks), 'flaggedVariants': len(flags), 'neurons': len(nodes)}), flush=True)
    with ThreadPoolExecutor(max_workers=a.threads) as pool:
        for index, (key, slot, info, record) in enumerate(pool.map(encode, tasks)):
            node = nodes[key]
            node['ours'][slot] = {'1024': f'/media/{record["sha256"]}.webp'}
            node['variantInfo'][slot] = info
            node['pending'].pop(slot, None)
            files.append(record)
            if (index+1) % 200 == 0:
                print(json.dumps({'encoded': index+1, 'total': len(tasks), 'seconds': round(time.monotonic()-started,1)}), flush=True)
    def save(key, value):
        raw = compact(value)
        file = a.output / key
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(raw)
        record = {'key': key, 'sha256': sha(raw), 'bytes': len(raw)}
        files.append(record)
        return record
    preserved = set(base_nodes) - updated_keys
    for key in preserved:
        nodes[key] = base_nodes[key]
        files.extend(reuse_image(info) for info in nodes[key]['variantInfo'].values())
    catalogue = {}
    fully_valid = complete_count = zero_diversity = 0
    for key, node in nodes.items():
        for slot in SLOTS:
            if slot not in node['ours'] and slot not in node['pending']:
                node['pending'][slot] = 'not_completed'
        if 'positive' in node['variantInfo']:
            node['config'] = node['variantInfo']['positive']['config']
        bits = ''.join('1' if slot in node['ours'] else 'q' if node['pending'].get(slot)=='quality_review' else '0' for slot in SLOTS)
        catalogue[key] = bits
        complete_count += bits == '11111'
        fully_valid += bits == '11111' and all(
            activation_valid(slot, node['variantInfo'][slot]) for slot in SLOTS)
        zero_diversity += 'diversity' in node['variantInfo'] and any(
            v == 0 for v in node['variantInfo']['diversity']['final']['score'])
        save(f'releases/{release}/nodes/{node["layer"]}/{node["channel"]}.json', node)
    image_count = sum(info['batchSize'] for node in nodes.values() for info in node['variantInfo'].values())
    summary = {'release': release, 'snapshotAt': now, 'variants': sum(len(n['ours']) for n in nodes.values()), 'images': image_count,
               'neurons': len(nodes), 'completeFiveModeNeurons': complete_count,
               'fullyValidFiveModeNeurons': fully_valid,
               'zeroActivationDiversityNeurons': zero_diversity,
               'zeroActivationPolicy': 'include_all' if a.include_zero_activations else 'include_diversity' if a.include_zero_diversity else 'reject',
               'qualityFlags': flags, 'protocol_sha256': sha(protocol_raw),
               'acceptedRepairVariants': repair_count}
    if a.complete_neurons_only:
        summary.update({'updatePolicy': 'complete_five_modes_only', 'updatedCompleteNeurons': len(updated_keys),
                        'skippedIncompleteNeurons': discovered - len(updated_keys), 'preservedExistingNeurons': len(preserved),
                        'baseRelease': base_manifest['summary']['release'] if base_manifest else None})
        for key in updated_keys:
            assert catalogue[key] == '11111', key
        for key in preserved:
            assert nodes[key] == base_nodes[key], key
        save('selection.json', {'updated': sorted(updated_keys), 'preserved': sorted(preserved)})
        files.pop()  # Local audit record, not a public media API object.
    save(f'releases/{release}/catalog.json', {**summary, 'nodes': catalogue})
    save(f'releases/{release}/manifest.json', summary)
    unique = {f['key']: f for f in files}
    (a.output / 'upload-manifest.json').write_bytes(compact({'summary': summary, 'objects': list(unique.values())}))
    (a.output / 'complete.json').write_bytes(compact({**summary, 'objects': len(unique), 'bytes': sum(f['bytes'] for f in unique.values())}))
    print(json.dumps({k: v for k,v in summary.items() if k!='qualityFlags'}), flush=True)

if __name__ == '__main__':
    main()
