"""Publication directory of reproduced and reference neuron reports."""
import argparse
import html
import json
import math
import re
import statistics
from pathlib import Path

TEMPLATE = Path(__file__).with_name('comparison_assets') / 'directory.html'


def _score_value(score):
    value = score.get('score') if score else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def _score_cell(score, name):
    value = _score_value(score)
    if value is None:
        return '<span class="unavailable" title="' + name + ' activation measurements not available">—</span>'
    parts = []
    if score.get('positive_ratio') is not None:
        parts.append(f"{name} max: {score['reproduced_max']:g} / {score['reference_max']:g}")
    if score.get('negative_ratio') is not None:
        parts.append(f"{name} min: |{score['reproduced_min']:g} / {score['reference_min']:g}|")
    note = html.escape('; '.join(parts), quote=True)
    return f'<span class="metric" title="{note}">{value:.3f}</span>'


def _score_header(rows, kind):
    values = [_score_value(row.get(kind + '_score')) for row in rows]
    values = [value for value in values if value is not None]
    parts = [kind.capitalize() + ' score']
    for label, aggregate in [('Mean', statistics.mean), ('Median', statistics.median)]:
        value = f'{aggregate(values):.3f}' if values else '—'
        parts.append(f'<span class="column-stat" title="{label} of {len(values)} available scores">'
                     f'{label} {value}</span>')
    return '<th>' + ''.join(parts) + '</th>'


def render(rows, *, corpus=''):
    """Render measured activation scores and their mean and median."""
    rows = list(rows)
    body = []
    for row in rows:
        neuron = row['neuron']
        if not re.fullmatch(r'L\d+_N\d+', neuron):
            raise ValueError('Expected neuron ID L<layer>_N<index>')
        link = (f'<a href="{neuron}/?compare=1">{neuron}</a>'
                if row.get('ready', True) else neuron)
        ours = html.escape(row.get('ours') or 'Report in progress')
        reference = html.escape(row.get('reference') or 'Reference explanation not provided')
        if not row.get('reference'):
            url = f'https://already-clean.goodfire.com/neuron-scientist/{neuron}/'
            reference = f'<a href="{url}">{reference}</a>'
        scores = ''.join('<td>' + _score_cell(row.get(kind + '_score'), kind.capitalize()) + '</td>'
                         for kind in ('blind', 'search'))
        body.append(f'<tr><td class="neuron">{link}</td><td>{ours}</td><td>{reference}</td>{scores}</tr>')
    models = list(dict.fromkeys(row['model'] for row in rows if row.get('model')))
    context = ' · '.join(models + ([corpus] if corpus else []))
    values = {'ROWS': '\n'.join(body), 'COUNT': str(len(rows)),
              'CONTEXT': html.escape(' · ' + context if context else ''),
              'SCORE_HEADERS': ''.join(_score_header(rows, kind) for kind in ('blind', 'search'))}
    return re.sub(r'\{\{(ROWS|COUNT|CONTEXT|SCORE_HEADERS)\}\}',
                  lambda match: values[match[1]], TEMPLATE.read_text())


def comparison_rows(reports, neurons, references=None):
    """Read completed exports; reference text remains outside research inputs."""
    from report import source_bundle
    from comparison import blind_score, search_score
    reports = Path(reports)
    rows = []
    for neuron in neurons:
        if not re.fullmatch(r'L\d+_N\d+', neuron):
            raise ValueError('Expected neuron ID L<layer>_N<index>')
        page = reports / neuron / 'index.html'
        reference_page = reports / neuron / 'comparison/original.html'
        if references:
            reference_page = Path(references) / (neuron + '.html')
        row = {'neuron': neuron, 'ready': page.is_file(), 'ours': None, 'reference': None,
               'blind_score': None, 'search_score': None}
        reference_bundle = source_bundle(reference_page.read_text())[0] if reference_page.is_file() else {}
        row['reference'] = (reference_bundle.get('claim') or {}).get('title')
        if page.is_file():
            bundle = source_bundle(page.read_text())[0]
            row['ours'] = (bundle.get('claim') or {}).get('title')
            row['model'] = (bundle.get('neuron') or {}).get('model')
            row['blind_score'] = blind_score(bundle, reference_bundle)
            row['search_score'] = search_score(bundle, reference_bundle)
        rows.append(row)
    return rows


def write_directory(output, rows, **kwargs):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.directory.tmp')
    temporary.write_text(render(rows, **kwargs))
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', required=True, type=Path, help='Saved directory-data.json report rows')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--corpus', default='')
    args = parser.parse_args()
    rows = json.loads(args.rows.read_text())
    write_directory(args.output, rows, corpus=args.corpus)


if __name__ == '__main__':
    main()
