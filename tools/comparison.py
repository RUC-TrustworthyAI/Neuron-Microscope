"""Human-only original/independent report comparison, added at publication time."""
import argparse
import hashlib
import html
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


def install(report, neuron, references=None):
    """Never call inside an agent stage or copy the reference into agent handoffs."""
    if not re.fullmatch(r'L\d+_N\d+', neuron):
        raise ValueError('Expected neuron ID L<layer>_N<index>')
    references = Path(references) if references else ROOT / 'data/reference-pages'
    reference = references / (neuron + '.html')
    if not reference.is_file():
        return False
    report = Path(report)
    folder = report / 'comparison'
    folder.mkdir(exist_ok=True)
    viewer_name = 'viewer.' + hashlib.sha256((ASSETS / 'viewer.js').read_bytes()).hexdigest()[:12] + '.js'
    _replace_file(ASSETS / 'viewer.js', folder / viewer_name)
    _replace_file(ASSETS / 'style.css', folder / 'style.css')
    _replace_file(reference, folder / 'original.html')
    page = report / 'index.html'
    source = MARKER.sub('', page.read_text())
    markup = (
        '<!-- neuron-comparison:start -->\n'
        '<link rel="stylesheet" href="comparison/style.css">\n'
        '<script defer src="comparison/' + viewer_name + '" data-neuron="' + html.escape(neuron, quote=True)
        + '" data-reference="comparison/original.html" data-original="https://already-clean.goodfire.com/neuron-scientist/'
        + neuron + '/"></script>\n'
        '<!-- neuron-comparison:end -->\n'
    )
    # Atomic replacement: publication trees may hardlink the unmodified scientific HTML.
    temporary = page.with_name('index.comparison.tmp')
    temporary.write_text(source.replace('</head>', markup + '</head>', 1))
    temporary.replace(page)
    return True


def publish(source, destination, neuron, references=None):
    """Add display controls only to the exported copy, never to model-facing artifacts."""
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
    return install(destination, neuron, references)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--neuron', required=True)
    parser.add_argument('--references', type=Path)
    args = parser.parse_args()
    comparison = publish(args.source, args.output, args.neuron, args.references)
    print(str(args.output / 'index.html'))
    print('Original comparison available: ' + str(comparison))


if __name__ == '__main__':
    main()
