"""Attach a verified RF snapshot to a portable static microscope export."""
import base64
import hashlib
from pathlib import Path
import re
import shutil

from .catalog import validate_target
from .site_export import read, within, write


def attach(output, snapshot):
    output, snapshot = Path(output), Path(snapshot)
    manifest = read(snapshot/'upload-manifest.json')
    summary = manifest['summary']
    release = summary['release']
    if summary.get('kind') != 'examples' or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', release):
        raise ValueError('Expected an RF examples snapshot')
    records = {item['key']: item for item in manifest['objects']}
    if len(records) != len(manifest['objects']):
        raise ValueError('Duplicate snapshot objects')
    for key, item in records.items():
        raw = within(snapshot, key).read_bytes()
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError(f'RF snapshot checksum mismatch: {key}')
    catalog_key = f'examples/{release}/catalog.json'
    if catalog_key not in records:
        raise ValueError('Missing RF catalog checksum')
    catalog = read(within(snapshot, catalog_key))
    if catalog.get('release') != release:
        raise ValueError('RF catalog release mismatch')
    nodes = catalog['nodes']
    if summary['neurons'] != len(nodes) or summary['samples'] != len(nodes)*48:
        raise ValueError('RF snapshot count mismatch')
    copied = set()
    for identity in nodes:
        layer, channel = identity.split(':')
        validate_target(layer, [int(channel)])
        key = f'examples/{release}/nodes/{layer}/{channel}.json'
        if key not in records:
            raise ValueError(f'Missing RF node: {identity}')
        node = read(within(snapshot, key))
        if node['layer'] != layer or node['channel'] != int(channel):
            raise ValueError('RF node identity mismatch')
        buckets = node['exampleBuckets']
        if [len(b['examples']) for b in buckets] != [16, 8, 8, 16]:
            raise ValueError('RF snapshot must contain all four sample buckets')
        heat = base64.b64decode(node['heatBase64'], validate=True)
        for bucket in buckets:
            for sample in bucket['examples']:
                height, width = sample['heatShape']
                offset = sample['heatOffset']
                if min(height, width) <= 0 or offset < 0 or offset+height*width > len(heat):
                    raise ValueError('Invalid RF heatmap offset')
                url = sample['atlasUrl']
                if match := re.fullmatch(r'/media/([a-f0-9]{64})\.webp', url):
                    digest = match[1]
                    image_key = f'images/{digest}.webp'
                    if image_key not in records or records[image_key]['sha256'] != digest:
                        raise ValueError('Missing content-addressed RF image')
                    relative = f'rf/images/{digest}.webp'
                    if digest not in copied:
                        destination = within(output, relative)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(within(snapshot, image_key), destination)
                        copied.add(digest)
                elif re.fullmatch(r'/samples/atlases/[0-9]+\.webp', url):
                    relative = url.lstrip('/')
                    if not within(output, relative).is_file():
                        raise ValueError('Full-input RF nodes require their original sample atlas')
                else:
                    raise ValueError(f'Unexpected RF atlas URL: {url}')
                sample['atlasUrl'] = relative
        destination = within(output, f'rf/nodes/{layer}/{channel}.json')
        destination.parent.mkdir(parents=True, exist_ok=True)
        write(destination, node)
    return {'release': release, 'nodeRoot': 'rf/nodes/', 'nodes': list(nodes),
            'neurons': len(nodes), 'samples': len(nodes)*48, 'policy': summary['policy']}
