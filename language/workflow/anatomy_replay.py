"""Render a same-template Anatomy comparison from saved native traces."""
from .anatomy_pairs import select_anatomy_pair
from .state import lock, read, write


def replay(source, root):
    from report import replace_anatomy_pair, source_bundle

    source, root = source.resolve(), root.resolve()
    if root == source or source in root.parents or root in source.parents:
        raise ValueError('Anatomy replay requires a separate output directory')
    if read(source / 'status.json')['phase'] != 'complete':
        raise ValueError('Anatomy replay requires a completed source investigation')

    config = read(source / 'config.json')
    artifacts = read(source / 'artifacts.json')
    source_html = (source / 'report/index.html').read_text(encoding='utf-8')
    source_data, _ = source_bundle(source_html)
    if source_data['neuron']['id'] != config['neuron']:
        raise ValueError('Source report and investigation identify different neurons')
    preferred = (source_data.get('display_selection') or {}).get('headline_figure_id')
    pair = select_anatomy_pair(artifacts['figures'], artifacts['all_measured'], preferred_figure_id=preferred)
    if pair is None:
        raise ValueError('Source investigation has no eligible measured Anatomy pair')
    page, bundle = replace_anatomy_pair(source_html, pair)

    receipt = {'source_investigation': str(source), 'neuron': config['neuron']}
    marker = root / 'anatomy-source.json'
    with lock(root / 'execution.lock', blocking=False):
        if marker.exists():
            previous = read(marker)
            if any(previous.get(key) != value for key, value in receipt.items()):
                raise ValueError('Output belongs to another investigation; choose a separate output directory')
        elif any(path.name != 'execution.lock' for path in root.iterdir()):
            raise ValueError('Anatomy replay requires an empty or matching output directory')
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
