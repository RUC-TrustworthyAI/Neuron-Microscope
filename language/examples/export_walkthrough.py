"""Export a recorded investigation as a local walkthrough."""
import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
from pathlib import Path, PureWindowsPath


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')


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
                if key not in ('model_path', 'config_path', 'settings_file', 'output', 'folder',
                               'evidence_file', 'error', 'dataset', 'model_snapshot')}
    return value


def recorded_request(folder, saved):
    accepted, input_id = saved.get('attempt'), saved.get('input_id')
    for path in sorted(folder.glob('attempt-*/request.json'), reverse=True):
        if accepted is not None:
            if path.parent.name == accepted:
                return read(path)
            continue
        request = read(path)
        if input_id is None:
            return request
        completion = path.with_name('completion.json')
        if not completion.is_file() or path.with_name('format-error.json').exists():
            continue
        saved_completion = read(completion)
        if saved_completion.get('status') == 'failed' or saved_completion.get('finish_reason') not in ('stop', 'end_turn'):
            continue
        messages = json.dumps(request.get('messages', request.get('input'))[:2], sort_keys=True).encode()
        if hashlib.sha256(messages).hexdigest() == input_id:
            return request
    return None


def export_calls(source, output, scope='', *, omit=(), omit_groups=()):
    calls = []
    for result_path in sorted(source.rglob('result.json')):
        folder = result_path.parent
        name = folder.relative_to(source).as_posix()
        if name in omit or name.partition('/')[0] in omit_groups:
            continue
        saved = read(result_path)
        request = recorded_request(folder, saved)
        if request is None:
            continue
        messages = request.get('messages', request.get('input'))
        payload = json.loads(messages[1]['content'])
        key = f'{scope}/{name}' if scope else name
        effort = request.get('reasoning_effort', request.get('reasoning', {}).get('effort'))
        item = {'id': key, 'model': request['model'], 'reasoning_effort': effort,
                'prompt': messages[0]['content'], 'input_keys': list(payload),
                'input_excerpt': preview(payload), 'output': saved['output']}
        if request.get('text', {}).get('format', {}).get('type') == 'text':
            item['response_format'] = 'text'
        if messages[2:]:
            item['correction_messages'] = [{'role': row['role'], 'content': row['content']} for row in messages[2:]]
        if 'points' in payload:
            item['input_count'] = len(payload['points'])
        if 'documents' in payload:
            item['context_count'] = len(payload['documents'])
        if 'prediction_feedback' in payload:
            feedback = payload['prediction_feedback']
            item['feedback_counts'] = {name: len(feedback.get(name, [])) for name in ('wrong_direction', 'strong_no_visible_cue')}
        path = Path('content/calls') / scope / (name.replace('/', '--') + '.json')
        write(output / path, item)
        calls.append({'id': key, 'file': path.as_posix(),
                      **{key: item[key] for key in ('input_count', 'context_count', 'feedback_counts') if key in item}})
    return calls


def report_script_block(page, identity):
    return re.search(r'<script\b[^>]*\bid=["\x27]' + re.escape(identity) + r'["\x27][^>]*>(.*?)</script>', page, re.S)


def report_script(page, identity):
    match = report_script_block(page, identity)
    return json.loads(match[1]) if match else None


def replace_script_body(page, block, body):
    return page[:block.start(1)] + body + page[block.end(1):]


def script_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace(
        '<', r'\u003c').replace('\u2028', r'\u2028').replace('\u2029', r'\u2029')


def refresh_synthesis(page):
    block = report_script_block(page, 'report-synthesis')
    if block is None:
        return page
    synthesis = json.loads(block[1])
    for summary in synthesis['section_prose'].values():
        summary.pop('evidence_ids', None)
    cited = dict.fromkeys(identity for finding in synthesis['claim']['findings'].values()
                          for identity in finding['evidence_ids'])
    synthesis['evidence'] = {identity: synthesis['evidence'][identity] for identity in cited}
    page = replace_script_body(page, block, script_json(synthesis))
    renderer = report_script_block(page, 'report-synthesis-render')
    if renderer is not None:
        source = Path(__file__).resolve().parents[1] / 'tools/report_synthesis.js'
        page = replace_script_body(page, renderer, source.read_text(encoding='utf-8'))
    return page


def portable_path(value):
    if isinstance(value, str) and (value.startswith('/') or PureWindowsPath(value).is_absolute()):
        return value.replace('\\', '/').rstrip('/').rsplit('/', 1)[-1]
    return value


def portable_metadata(value):
    if isinstance(value, dict):
        return {key: portable_path(item) if key in ('archive', 'archive_model', 'model', 'path')
                and isinstance(item, str) else portable_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_metadata(item) for item in value]
    return value


def portable_report(page):
    block = report_script_block(page, 'neuron-labels')
    if block is not None:
        labels = json.loads(block[1])
        scope = portable_metadata(labels['scope'])
        if scope != labels['scope']:
            labels['scope'] = scope
            page = replace_script_body(page, block, script_json(labels))
    return refresh_synthesis(page)


def report_replay(page, bundle):
    fields = {}
    route_prediction = (bundle.get('tests') or {}).get('route_prediction')
    if route_prediction is not None:
        fields['route_prediction'] = route_prediction
    labels = report_script(page, 'neuron-labels')
    if labels is not None:
        fields['neuron_labels'] = {'labels': labels['labels'], 'scope': labels['scope']}
    synthesis = report_script(page, 'report-synthesis')
    if synthesis is not None:
        fields['synthesis'] = synthesis
    return fields


def followup_replay(path):
    ledger = read(path)
    for current in ledger['rounds']:
        for investigation in current['investigations']:
            investigation.pop('evidence_file', None)
            investigation.pop('error', None)
            evidence = investigation.get('evidence', {})
            if 'source' in evidence:
                evidence['source'] = {key: value for key, value in evidence['source'].items()
                                      if key in ('kind', 'neuron')}
    ledger['report'] = 'reports/final.html#criticism'
    return ledger


def write_report(reports, name, source):
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / (name + '.html.gz')).open('wb') as output:
        with gzip.GzipFile(filename='', fileobj=output, mode='wb', compresslevel=6, mtime=0) as compressed:
            compressed.write(source.encode('utf-8'))
    loader = '''<!doctype html><meta charset="utf-8"><title>Neuron report</title>
<p style="font:15px system-ui;padding:32px;color:#777">Loading the recorded report…</p><script>
(async()=>{try{const r=await fetch(PAYLOAD);if(!r.ok)throw Error('Could not load the report.');let html=await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).text();document.open();document.write(html);document.close();}catch(e){document.body.textContent=e.message}})();</script>'''
    (reports / (name + '.html')).write_text(loader.replace('PAYLOAD', json.dumps(name + '.html.gz')), encoding='utf-8')


def archive_path(root, path):
    target = root / path
    if not target.is_relative_to(root) or target.resolve() != target:
        raise ValueError(f'Invalid archive path: {path}')
    return target


def export_archive(archive, output):
    archive, output = archive.resolve(), output.resolve()
    manifest = Path('content/episode.json')
    episode = read(archive_path(archive, manifest))
    destination = archive_path(output, manifest)
    calls = []
    for call in episode['calls']:
        path = Path(call['file'])
        if path.parts[:2] != ('content', 'calls') or len(path.parts) < 3:
            raise ValueError(f'Invalid archive path: {path}')
        calls.append((archive_path(archive, path), archive_path(output, path)))
    reports = {}
    for name in ('evidence', 'final'):
        loader = Path('reports', f'{name}.html')
        compressed = loader.with_suffix('.html.gz')
        reports[name] = archive_path(archive, compressed)
        archive_path(output, compressed)
        archive_path(output, loader)
    for source, target in calls:
        write(target, read(source))
    for name, source in reports.items():
        with gzip.open(source, 'rt', encoding='utf-8', newline='') as report:
            page = portable_report(report.read())
        if name == 'final':
            episode.update(report_replay(page, report_script(page, 'bundle')))
        write_report(output / 'reports', name, page)
    write(destination, episode)


def export(run, output):
    config = read(run / 'config.json')
    initial_report = Path(config['report'])
    report_data = read(initial_report / 'report.json')
    source_identity = portable_metadata(report_data.get('identity', {}))
    artifacts = read(run / 'artifacts.json')
    history = read(run / 'interpretation-history.json')
    sample = read(run / 'sample/sections.json')['spectrum']
    sampling = read(run / 'sample/sampling.json')
    evidence = read(run / 'evidence.json')
    final_html = portable_report((run / 'report/index.html').read_text(encoding='utf-8'))
    bundle = report_script(final_html, 'bundle')
    followup_path = run / 'followup/investigation.json'
    status_path = run / 'status.json'
    followup = followup_path.is_file() and (
        any(row.get('investigation_id') for row in bundle.get('criticism', ()))
        or (status_path.is_file() and read(status_path).get('stop_reason')))
    replay = report_replay(final_html, bundle)
    active_api = run / ('followup/api' if followup else 'api')
    claim = replay.get('synthesis', {}).get('claim', {})
    overview = active_api / 'report-synthesis/overview'
    split_synthesis = all(
        key in claim and (overview / key / 'result.json').is_file()
        and read(overview / key / 'result.json')['output'].get('text') == claim[key]
        for key in ('title', 'lede')
    )
    retired = ('report-synthesis',) if split_synthesis else ()
    for name in ('content', 'reports', 'screenshots'):
        shutil.rmtree(output / name, ignore_errors=True)
    if followup:
        calls = export_calls(run / 'api', output, omit_groups=('report-synthesis',))
        calls.extend(export_calls(active_api, output, 'followup', omit=retired))
    else:
        calls = export_calls(active_api, output, omit=retired)

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
    headline = (bundle.get('display_selection') or {}).get('headline_figure_id')
    if not headline:
        head = bundle.get('headline') or {}
        frame = (head.get('comparison') or {}).get('template')
        headline = next((figure['id'] for figure in figures if figure.get('template') == frame), figures[0]['id'])
    search = read(run / 'search-progress.json')
    costs_path = run / 'report/interpretation-costs.json'
    costs = read(costs_path) if costs_path.exists() else bundle.get('interpretation_costs')
    episode = {
        'unit': config['neuron'], 'subject_model': source_identity.get('model') or portable_path(config['model']),
        'dataset': 'Saved native activation corpus',
        'source_identity': source_identity,
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
    episode.update(replay)
    if followup:
        episode['followup'] = followup_replay(followup_path)
    write(output / 'content/episode.json', episode)
    reports = output / 'reports'
    source_root = os.path.commonpath([config[key] for key in ('model', 'report', 'activations')])
    evidence_html = portable_report((initial_report / 'index.html').read_text(encoding='utf-8'))
    for name, source in [('evidence', evidence_html), ('final', final_html)]:
        source = re.sub(r'<!-- neuron-comparison:start -->.*?<!-- neuron-comparison:end -->\s*', '', source, flags=re.S)
        if source_root != '/':
            source = re.sub(re.escape(source_root) + r'/[^"\s<>\\]+', lambda m: Path(m[0]).name, source)
        write_report(reports, name, source)
    print(json.dumps({'rounds': len(rounds), 'calls': len(calls), 'measured_texts': len(measured_rows),
                      'selected_round': selected['round'], 'points': [len(row['points']) for row in rounds]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run', type=Path, help='completed investigation directory')
    source.add_argument('--archive', type=Path, help='existing static walkthrough directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.archive:
        export_archive(args.archive.resolve(), args.output.resolve())
    else:
        export(args.run.resolve(), args.output.resolve())
