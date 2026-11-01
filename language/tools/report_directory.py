"""Publication-only directory of reproduced and reference interpretations."""
import argparse
import hashlib
import html
import json
import re
from pathlib import Path

TEMPLATE = Path(__file__).with_name('comparison_assets') / 'directory.html'


def render(rows, *, corpus='', model='deepseek-v4.1-flash-1017', reasoning='high'):
    """Render saved comparison judgments without calling an LLM."""
    esc = html.escape
    body = []
    for row in rows:
        neuron = row['neuron']
        if not re.fullmatch(r'L\d+_N\d+', neuron):
            raise ValueError('Expected neuron ID L<layer>_N<index>')
        link = (f'<a href="{neuron}/?compare=1">{neuron}</a>'
                if row.get('ready', True) else neuron)
        ours = esc(row.get('ours') or 'Report in progress')
        reference = esc(row.get('reference') or 'Reference explanation not provided')
        if not row.get('reference'):
            url = f'https://already-clean.goodfire.com/neuron-scientist/{neuron}/'
            reference = f'<a href="{url}">{reference}</a>'
        judgment = row.get('judgment')
        if judgment:
            names = {'A': 'Reproduced report' if row['comparison_a'] == 'reproduced' else 'Reference report',
                     'B': 'Reference report' if row['comparison_a'] == 'reproduced' else 'Reproduced report'}
            reason = re.sub(r'\b[AB]\b', lambda m: names[m[0]], judgment['reason'])
            value = judgment['semantic_similarity']
            score = (f'<span class="score" tabindex="0" aria-label="{value} out of 5. {esc(reason, quote=True)}">'
                     f'{value}<span class="score-note">{esc(reason)}</span></span>')
        else:
            note = row.get('score_status') or ('Awaiting similarity score' if row.get('ready', True) else 'Report in progress')
            score = f'<span class="unavailable" title="{esc(note, quote=True)}">—</span>'
        blind = row.get('blind_score')
        if blind is None:
            blind_html = '<span class="unavailable" title="Blind activation measurements not available">—</span>'
        else:
            parts = []
            if blind.get('positive_ratio') is not None:
                parts.append(f"Blind max: {blind['reproduced_max']:g} / {blind['reference_max']:g}")
            if blind.get('negative_ratio') is not None:
                parts.append(f"Blind min: |{blind['reproduced_min']:g} / {blind['reference_min']:g}|")
            note = '; '.join(parts)
            blind_html = f'<span class="score" title="{esc(note, quote=True)}">{blind["score"]:.3f}</span>'
        body.append(f'<tr><td class="neuron">{link}</td><td>{ours}</td><td>{reference}</td>'
                    f'<td class="similarity">{score}</td><td class="similarity">{blind_html}</td></tr>')
    values = {'ROWS': '\n'.join(body), 'CORPUS': esc(' · ' + corpus if corpus else ''),
              'MODEL': esc(model), 'REASONING': esc(reasoning)}
    return re.sub(r'\{\{(ROWS|CORPUS|MODEL|REASONING)\}\}', lambda m: values[m[1]], TEMPLATE.read_text())


def comparison_rows(reports, neurons, references=None, records=None):
    """Read completed exports; reference text remains outside research inputs."""
    from report import source_bundle
    from comparison import blind_score
    reports = Path(reports)
    rows = []
    for neuron in neurons:
        page = reports / neuron / 'index.html'
        reference_page = reports / neuron / 'comparison/original.html'
        if references:
            reference_page = Path(references) / (neuron + '.html')
        row = {'neuron': neuron, 'ready': page.is_file(), 'ours': None, 'reference': None, 'judgment': None}
        reference_bundle = source_bundle(reference_page.read_text())[0] if reference_page.is_file() else {}
        reference_claim = reference_bundle.get('claim')
        if reference_claim:
            row['reference'] = reference_claim.get('title')
        if page.is_file():
            bundle = source_bundle(page.read_text())[0]
            ours = bundle['claim']
            row['blind_score'] = blind_score(bundle, reference_bundle)
            row['ours'] = ours.get('title')
            if reference_claim:
                a_is_ours = int(hashlib.sha256(neuron.encode()).hexdigest(), 16) % 2 == 0
                row['comparison_a'] = 'reproduced' if a_is_ours else 'reference'
                row['input'] = {'A': ours if a_is_ours else reference_claim,
                                'B': reference_claim if a_is_ours else ours}
                saved = Path(records) / 'comparisons' / (neuron + '-judgment.json') if records else None
                if saved and saved.is_file():
                    result = json.loads(saved.read_text())
                    if result.get('input') == row['input']:
                        row['judgment'] = result['judgment']
            else:
                row['score_status'] = 'Reference explanation not provided'
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
    parser.add_argument('--rows', required=True, type=Path, help='Saved directory-data.json comparison rows')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--corpus', default='')
    parser.add_argument('--model', default='deepseek-v4.1-flash-1017')
    parser.add_argument('--reasoning', default='high')
    parser.add_argument('--score', action='store_true', help='Score new completed explanation pairs before rendering')
    parser.add_argument('--records', type=Path, help='Local API and judgment records for --score')
    args = parser.parse_args()
    rows = json.loads(args.rows.read_text())
    if args.score:
        if args.records is None:
            parser.error('--score requires --records')
        from directory_scores import score_rows
        score_rows(rows, args.records)
        args.rows.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    write_directory(args.output, rows, corpus=args.corpus, model=args.model, reasoning=args.reasoning)


if __name__ == '__main__':
    main()
