"""Export side-by-side reference and reproduction reports."""
import argparse
import hashlib
import html
import math
import os
import re
import shutil
from pathlib import Path

ROOT = Path(os.environ.get('NEURON_MICROSCOPE_ROOT', Path(__file__).resolve().parents[1]))
ASSETS = Path(__file__).with_name('comparison_assets')
MARKER = re.compile(r'<!-- neuron-comparison:start -->.*?<!-- neuron-comparison:end -->\s*', re.S)


def _copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return str(target)


def _replace_file(source, target):
    temporary = target.with_name(target.name + '.comparison.tmp')
    shutil.copy2(source, temporary)
    temporary.replace(target)


def blind_score(reproduced, reference):
    """Compare achieved blind-test extrema in two completed reports."""
    def extrema(bundle):
        tests = bundle.get('tests') or {}
        return tuple((tests.get(field) or {}).get('achieved') for field in ('blind_max', 'blind_min'))
    return _extrema_score(extrema(reproduced), extrema(reference))


def search_score(reproduced, reference):
    """Compare measured activation-search extrema in two completed reports."""
    def extrema(bundle):
        search = bundle.get('search') or {}
        return search.get('best_positive'), search.get('best_negative')
    return _extrema_score(extrema(reproduced), extrema(reference))


def _extrema_score(reproduced, reference):
    values, ratios = {}, {}
    for pole, ours, theirs in zip(('positive', 'negative'), reproduced, reference):
        values[pole] = (ours, theirs)
        if ours is None or theirs is None:
            continue
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in (ours, theirs)) or theirs == 0:
            return None
        ratio = ours / theirs
        ratios[pole] = ratio if pole == 'positive' else abs(ratio)
    if not ratios:
        return None
    maximum, minimum = values['positive'], values['negative']
    return {'score': sum(ratios.values()) / len(ratios),
            'basis': 'both' if len(ratios) == 2 else next(iter(ratios)),
            'positive_ratio': ratios.get('positive'), 'negative_ratio': ratios.get('negative'),
            'reproduced_max': maximum[0], 'reference_max': maximum[1],
            'reproduced_min': minimum[0], 'reference_min': minimum[1]}


def install(report, neuron, references=None, directory='../'):
    """Install comparison controls in an exported report."""
    if not re.fullmatch(r'L\d+_N\d+', neuron):
        raise ValueError('Expected neuron ID L<layer>_N<index>')
    references = Path(references) if references else ROOT / 'data/reference-pages'
    reference = references / (neuron + '.html')
    available = reference.is_file()
    report = Path(report)
    folder = report / 'comparison'
    folder.mkdir(exist_ok=True)
    viewer_name = 'viewer.' + hashlib.sha256((ASSETS / 'viewer.js').read_bytes()).hexdigest()[:12] + '.js'
    _replace_file(ASSETS / 'viewer.js', folder / viewer_name)
    _replace_file(ASSETS / 'style.css', folder / 'style.css')
    if available:
        _replace_file(reference, folder / 'original.html')
    page = report / 'index.html'
    source = MARKER.sub('', page.read_text())
    source = source.replace("['Neuron paper',", "['Neuron report',")
    attributes = '" data-directory="' + html.escape(directory, quote=True)
    if available:
        attributes += ('" data-reference="comparison/original.html" data-original="'
                       'https://already-clean.goodfire.com/neuron-scientist/' + neuron + '/')
        from report import source_bundle
        score = blind_score(source_bundle(source)[0], source_bundle(reference.read_text())[0])
        if score is not None:
            attributes += '" data-blind-score="' + str(score['score'])
            attributes += '" data-blind-score-basis="' + score['basis']
    markup = (
        '<!-- neuron-comparison:start -->\n'
        '<link rel="stylesheet" href="comparison/style.css">\n'
        '<script defer src="comparison/' + viewer_name + '" data-neuron="' + html.escape(neuron, quote=True)
        + attributes + '"></script>\n'
        '<!-- neuron-comparison:end -->\n'
    )
    # Atomic replacement: publication trees may hardlink the unmodified scientific HTML.
    temporary = page.with_name('index.comparison.tmp')
    temporary.write_text(source.replace('</head>', markup + '</head>', 1))
    temporary.replace(page)
    return available


def publish(source, destination, neuron, references=None, directory='../'):
    """Create an exported report with comparison controls."""
    source, destination = Path(source), Path(destination)
    shutil.copytree(source, destination, copy_function=_copy)
    from exemplar_display import render as render_exemplars
    page = destination / 'index.html'
    source_html = page.read_text()
    rendered = render_exemplars(source_html)
    if rendered != source_html:
        temporary = page.with_name('index.display.tmp')
        temporary.write_text(rendered)
        temporary.replace(page)
    return install(destination, neuron, references, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--neuron', required=True)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--directory', default='../', help='Link from the report to its directory')
    args = parser.parse_args()
    comparison = publish(args.source, args.output, args.neuron, args.references, args.directory)
    print(str(args.output / 'index.html'))
    print('Original comparison available: ' + str(comparison))


if __name__ == '__main__':
    main()
