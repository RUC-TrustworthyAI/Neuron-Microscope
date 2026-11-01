"""Export a recorded investigation as a local walkthrough."""
import argparse
import gzip
import json
import os
import re
import shutil
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')))


def measured(row):
    peak = row.get('selected') or {}
    return {key: row[key] for key in ('id', 'label', 'text', 'expected', 'score', 'scored_positions') if key in row} | {
        'peak': {key: peak[key] for key in ('position', 'token', 'product', 'offset') if key in peak}}


def preview(value, limit=3):
    """Select readable evidence excerpts while preserving exact displayed values."""
    if isinstance(value, list):
        return {'total': len(value), 'examples': [preview(row, limit) for row in value[:limit]]} if len(value) > limit else [preview(row, limit) for row in value]
    if isinstance(value, dict):
        return {key: preview(row, limit) for key, row in value.items()
                if key not in ('model_path', 'config_path', 'settings_file', 'output', 'folder')}
    return value


def export(run, output):
    config = read(run / 'config.json')
    initial_report = Path(config['report'])
    artifacts = read(run / 'artifacts.json')
    history = read(run / 'interpretation-history.json')
    sample = read(run / 'sample/sections.json')['spectrum']
    sampling = read(run / 'sample/sampling.json')
    evidence = read(run / 'evidence.json')
    for name in ('content', 'reports', 'screenshots'):
        shutil.rmtree(output / name, ignore_errors=True)
    content = output / 'content'
    calls = []
    for result_path in sorted((run / 'api').rglob('result.json')):
        folder = result_path.parent
        attempts = sorted(p for p in folder.glob('attempt-*') if (p / 'request.json').exists())
        if not attempts:
            continue
        attempt = attempts[-1]
        request = read(attempt / 'request.json')
        messages = request['messages']
        payload = json.loads(messages[1]['content'])
        key = str(folder.relative_to(run / 'api'))
        item = {'id': key, 'model': request['model'], 'reasoning_effort': request.get('reasoning_effort'),
                'prompt': messages[0]['content'], 'input_keys': list(payload),
                'input_excerpt': preview(payload), 'output': read(result_path)['output']}
        if 'points' in payload:
            item['input_count'] = len(payload['points'])
        if 'documents' in payload:
            item['context_count'] = len(payload['documents'])
        if 'prediction_feedback' in payload:
            feedback = payload['prediction_feedback']
            item['feedback_counts'] = {name: len(feedback.get(name, [])) for name in ('wrong_direction', 'strong_no_visible_cue')}
        filename = key.replace('/', '--') + '.json'
        write(content / 'calls' / filename, item)
        calls.append({'id': key, 'file': 'content/calls/' + filename,
                      **{key: item[key] for key in ('input_count', 'context_count', 'feedback_counts') if key in item}})

    rounds = []
    for entry in history:
        raw = read(run / entry['explanation_path'])['output']
        predictions = read(run / entry['predictions_path'])
        points = []
        measured_points = {point['id']: point for point in sample['points']}
        for record in predictions['records']:
            point = measured_points[record['id']]
            points.append({key: record[key] for key in ('id', 'group', 'predicted_pole', 'reason', 'activation',
                                                       'true_pole', 'correct', 'anchor', 'observed_prefix', 'nearby_continuation') if key in record}
                          | {'bin': point['bin']})
        rounds.append({'round': entry['round'], 'selected': entry['selected'], 'explanation': raw,
                       'rules': predictions['response_rules'], 'summary': entry['summary'],
                       'points': points})
    selected = next(row for row in rounds if row['selected'])
    measured_rows = [measured(row) for row in artifacts['all_measured']]
    figures = artifacts['figures']
    headline = None
    final_html = (run / 'report/index.html').read_text()
    match = re.search(r'<script\b[^>]*\bid=["\x27]bundle["\x27][^>]*>(.*?)</script>', final_html, re.S)
    bundle = json.loads(match[1])
    headline = (bundle.get('display_selection') or {}).get('headline_figure_id')
    if not headline:
        head = bundle.get('headline') or {}
        frame = (head.get('comparison') or {}).get('template')
        headline = next((figure['id'] for figure in figures if figure.get('template') == frame), figures[0]['id'])
    search = read(run / 'search-progress.json')
    costs_path = run / 'report/interpretation-costs.json'
    costs = read(costs_path) if costs_path.exists() else bundle.get('interpretation_costs')
    episode = {
        'unit': config['neuron'], 'subject_model': 'Qwen3-32B', 'dataset': 'Reasoning Mix 100M',
        'n_tokens': sampling['tokens_scanned'], 'corpus': artifacts['corpus'],
        'preparation': {key: sampling[key] for key in ('sampled_points', 'population_bin_counts', 'sample_bin_counts',
                       'scan_seconds', 'wall_seconds', 'new_model_forwards', 'gpu_seconds', 'llm_requests')},
        'spectrum': {'edges': sample['edges'], 'corpus_hist': sample['corpus_hist']['counts']},
        'rounds': rounds, 'selected_round': selected['round'], 'calls': calls,
        'evidence': {'statistics': evidence['statistics'], 'examples': evidence['samples'],
                     'documents': evidence['documents'][:3],
                     'token_selectivity': evidence.get('token_selectivity'),
                     'peak_census': evidence.get('peak_census'), 'logit_effect': evidence.get('logit_effect')},
        'search_progress': search, 'measurements': measured_rows, 'figures': figures, 'headline': headline,
        'controls_initial': read(run / 'measurements/figures-initial.json'),
        'controls_revised': read(run / 'measurements/figures-revised.json'),
        'blind': {key: value for key, value in artifacts['tests'].items() if key in ('blind_max', 'blind_min')},
        'atlas': [{key: row[key] for key in ('id', 'text', 'selected', 'upstream', 'downstream') if key in row} for row in artifacts['atlas']],
        'graphs': [{'id': row['id'], 'text': row['graph']['text'], 'position': row['graph']['position']} for row in artifacts['graphs']],
        'review': artifacts['review'], 'analysis': artifacts['revision'], 'costs': costs,
        'report': 'reports/final.html', 'initial_report': 'reports/evidence.html',
    }
    write(content / 'episode.json', episode)
    reports = output / 'reports'
    reports.mkdir(exist_ok=True)
    for name, source in [('evidence', (initial_report / 'index.html').read_text()), ('final', final_html)]:
        source = re.sub(r'<!-- neuron-comparison:start -->.*?<!-- neuron-comparison:end -->\s*', '', source, flags=re.S)
        source_root = os.path.commonpath([config[key] for key in ('model', 'report', 'activations')])
        if source_root != '/':
            source = re.sub(re.escape(source_root) + r'/[^"\s<>\\]+', lambda m: Path(m[0]).name, source)
        (reports / (name + '.html.gz')).write_bytes(gzip.compress(source.encode(), compresslevel=6, mtime=0))
        loader = '''<!doctype html><meta charset="utf-8"><title>Neuron report</title>
<p style="font:15px system-ui;padding:32px;color:#777">Loading the recorded report…</p><script>
(async()=>{try{const r=await fetch(PAYLOAD);if(!r.ok)throw Error('Could not load the report.');let html=await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).text();document.open();document.write(html);document.close();}catch(e){document.body.textContent=e.message}})();</script>'''
        (reports / (name + '.html')).write_text(loader.replace('PAYLOAD', json.dumps(name + '.html.gz')))
    print(json.dumps({'rounds': len(rounds), 'calls': len(calls), 'measured_texts': len(measured_rows),
                      'selected_round': selected['round'], 'points': [len(row['points']) for row in rounds]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.run.resolve(), args.output.resolve())
