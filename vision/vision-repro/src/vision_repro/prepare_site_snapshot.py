"""Create an immutable, CPU-only export of all numerically valid finished jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import time
from PIL import Image

from .site_export import public_metadata

SLOTS = ['positive', 'path_positive', 'diversity', 'negative', 'path_negative']
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=False)
    release = a.output.name
    now = datetime.now(timezone.utc).isoformat()
    protocol_raw = (a.root / 'protocol.json').read_bytes()
    protocol = json.loads(protocol_raw)
    for name, digest in protocol['source_sha256'].items():
        assert sha((Path(__file__).parent / name).read_bytes()) == digest
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
                continue
            tasks.append((key, slot, folder, i, members, meta, evaluation))
    def encode(task):
        key, slot, folder, i, members, meta, evaluation = task
        raw = (folder / f'items/{i}/final.png').read_bytes()
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
        info = {'batchSize': members, 'runId': f'{a.root.name}/batches/{folder.name}/items/{i}',
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
    catalogue = {}
    fully_valid = 0
    for key, node in nodes.items():
        for slot in SLOTS:
            if slot not in node['ours'] and slot not in node['pending']:
                node['pending'][slot] = 'not_completed'
        if 'positive' in node['variantInfo']:
            node['config'] = node['variantInfo']['positive']['config']
        bits = ''.join('1' if slot in node['ours'] else 'q' if node['pending'].get(slot)=='quality_review' else '0' for slot in SLOTS)
        catalogue[key] = bits
        fully_valid += bits == '11111'
        save(f'releases/{release}/nodes/{node["layer"]}/{node["channel"]}.json', node)
    image_count = sum(info['batchSize'] for node in nodes.values() for info in node['variantInfo'].values())
    summary = {'release': release, 'snapshotAt': now, 'variants': len(tasks), 'images': image_count,
               'neurons': len(nodes), 'fullyValidFiveModeNeurons': fully_valid,
               'qualityFlags': flags, 'protocol_sha256': sha(protocol_raw)}
    save(f'releases/{release}/catalog.json', {**summary, 'nodes': catalogue})
    save(f'releases/{release}/manifest.json', summary)
    unique = {f['key']: f for f in files}
    (a.output / 'upload-manifest.json').write_bytes(compact({'summary': summary, 'objects': list(unique.values())}))
    (a.output / 'complete.json').write_bytes(compact({**summary, 'objects': len(unique), 'bytes': sum(f['bytes'] for f in unique.values())}))
    print(json.dumps({k: v for k,v in summary.items() if k!='qualityFlags'}), flush=True)

if __name__ == '__main__':
    main()
