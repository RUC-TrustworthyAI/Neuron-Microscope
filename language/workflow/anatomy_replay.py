"""Render a same-template Anatomy comparison from saved native traces."""
import hashlib
import json

from .anatomy_pairs import select_anatomy_pair
from .state import lock, write


def replay(source, root):
    from report import replace_anatomy_pair, source_bundle

    source, root = source.resolve(), root.resolve()
    if root == source or source in root.parents or root in source.parents:
        raise ValueError('Anatomy replay requires a separate output directory')
    if json.loads((source / 'status.json').read_bytes())['phase'] != 'complete':
        raise ValueError('Anatomy replay requires a completed source investigation')

    files = {name: (source / name).read_bytes()
             for name in ('config.json', 'artifacts.json', 'report/index.html')}
    config = json.loads(files['config.json'])
    artifacts = json.loads(files['artifacts.json'])
    source_html = files['report/index.html'].decode('utf-8')
    source_data, _ = source_bundle(source_html)
    if source_data['neuron']['id'] != config['neuron']:
        raise ValueError('Source report and investigation identify different neurons')
    preferred = (source_data.get('display_selection') or {}).get('headline_figure_id')
    pair = select_anatomy_pair(artifacts['figures'], artifacts['all_measured'], preferred_figure_id=preferred)
    if pair is None:
        raise ValueError('Source investigation has no eligible measured Anatomy pair')
    page, bundle = replace_anatomy_pair(source_html, pair)

    receipt = {'source_investigation': str(source), 'neuron': config['neuron'],
               'files': {name: hashlib.sha256(blob).hexdigest() for name, blob in files.items()}}
    marker = root / 'anatomy-source.json'
    if root.is_dir() and not marker.exists() and any(path.name != 'execution.lock' for path in root.iterdir()):
        raise ValueError('Anatomy replay requires an empty or matching output directory')
    with lock(root / 'execution.lock', blocking=False):
        if marker.exists():
            if json.loads(marker.read_bytes()) != receipt:
                raise ValueError('Source investigation changed; choose a separate output directory')
        elif any(path.name != 'execution.lock' for path in root.iterdir()):
            raise ValueError('Anatomy replay requires an empty or matching output directory')
        else:
            write(marker, receipt)
        output = root / 'report'
        write(root / 'status.json', {'phase': 'rendering'})
        write(root / 'anatomy-pair.json', pair)
        write(output / 'paper-data.json', {'bundle': bundle})
        temporary = output / 'index.html.tmp'
        temporary.write_text(page, encoding='utf-8')
        temporary.replace(output / 'index.html')
        write(root / 'status.json', {'phase': 'complete', 'html': str(output / 'index.html')})
    return output
