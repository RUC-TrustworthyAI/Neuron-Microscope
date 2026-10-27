#!/usr/bin/env python3
"""Export report-only resources and selected prompt examples for the static demo."""
import argparse
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_SCRIPT = re.compile(r'(<script\b(?=[^>]*\btype=["\x27]application/json["\x27])(?=[^>]*\bid=["\x27]([^"\x27]+)["\x27])[^>]*>)(.*?)(</script>)', re.S)
LOCAL_PATH = re.compile(r'/(?:data|Users|home|mnt)/[^\s<>"\x27]+')
OMIT = {'api_sources', 'worker_sources', 'unassigned_api_records', 'source_files',
        'source_report', 'source_html_sha256', 'scan_source', 'reuse_measurements',
        'source_file', 'logit_arrays', 'model_snapshot', 'local_metadata'}
TEXT = {'text', 'observed_prefix', 'story', 'caption', 'description', 'prediction',
        'one_paragraph', 'negative_pole', 'lede', 'what_would_change_my_mind'}


def read(path):
    return json.loads(path.read_text())


def public(value, key=''):
    if isinstance(value, dict):
        return {k: public(v, k) for k, v in value.items() if k not in OMIT}
    if isinstance(value, list):
        return [public(v, key) for v in value]
    if isinstance(value, str) and key not in TEXT:
        return LOCAL_PATH.sub('[local path]', value)
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(public(value), ensure_ascii=False, indent=2))


def report(source):
    def replace(match):
        data = json.loads(match[3])
        if match[2] == 'bundle':
            # all_exemplars is an unused duplicate of the displayed example data.
            data.pop('all_exemplars', None)
        encoded = json.dumps(public(data), ensure_ascii=False, allow_nan=False,
                             separators=(',', ':')).replace('<', '\\u003c')
        return match[1] + encoded + match[4]
    result = JSON_SCRIPT.sub(replace, source)
    result = re.sub(r'<link\b[^>]*href=["\x27]comparison/[^>]*>', '', result)
    result = re.sub(r'<script\b[^>]*src=["\x27]comparison/[^>]*>.*?</script>', '', result, flags=re.S)
    return result


def loader(filename):
    return '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Neuron Microscope · Saved report</title>
<style>body{margin:0;font:14px/1.8 system-ui;color:#555;background:#fff}#loading{padding:48px;max-width:650px;margin:auto}a{color:#333}</style>
<div id="loading">Loading the saved interactive report…</div>
<script>
(async()=>{try{
 const response=await fetch('payloads/''' + filename + '''');
 if(!response.ok)throw new Error('The report resource is unavailable.');
 const stream=response.body.pipeThrough(new DecompressionStream('gzip'));
 const html=await new Response(stream).text();
 document.open();document.write(html);document.close();
}catch(error){document.getElementById('loading').textContent=error.message;}})();
</script></html>'''


def prompt_example(source, item):
    saved = read(source / item['file'])
    packet = saved.get('request', {})
    messages = packet.get('messages', [])
    system = next((m.get('content', '') for m in messages if m.get('role') == 'system'), '')
    raw = next((m.get('content', '') for m in messages if m.get('role') == 'user'), '')
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        payload = raw
    selected = []
    if isinstance(payload, dict):
        for field in ('points', 'cases', 'contexts'):
            if isinstance(payload.get(field), list):
                rows = payload[field]
                selected = sorted(rows, key=lambda p: len(json.dumps(p)))[:2]
                payload[field] = selected
        if 'experiments' in payload:
            payload['experiments'] = payload['experiments'][:2]
    response = saved.get('response_text', '')
    try:
        parsed = json.loads(response)
        if selected and isinstance(parsed, dict):
            ids = {str(row.get('id')) for row in selected}
            for field in ('predictions', 'assignments'):
                if isinstance(parsed.get(field), list):
                    parsed[field] = [r for r in parsed[field] if str(r.get('id')) in ids]
        response = json.dumps(public(parsed), ensure_ascii=False, indent=2)
    except (ValueError, TypeError):
        pass
    return {'format': 'selected-example', 'system': system,
            'input_excerpt': public(payload), 'output_excerpt': response}


def action_examples(trace):
    selected = []
    wanted = ('SKILL.md', 'read_evidence.py', 'run_experiments.py',
              'run_tests.py', 'submit_interpretation.py', 'complete_stage',
              'submit_revision', 'complete.py')
    used = set()
    for call in trace['calls']:
        args = call.get('arguments', {})
        command = args.get('command', '') if isinstance(args, dict) else str(args)
        kind = next((x for x in wanted if x in command), None)
        if kind is None or kind in used:
            continue
        # Retain a documented invocation, without shell bookkeeping or raw logs.
        lines = command.splitlines()
        line = next((x.strip() for x in lines if kind in x and ('python ' in x or 'cat ' in x)), '')
        if not line:
            continue
        line = line.split(' && ')[0].split(' > ')[0].split(' | ')[0]
        line = line.split(" <<'JSON'")[0].split(" <<'PY'")[0]
        if len(line) > 500:
            continue
        used.add(kind)
        summaries = {
            'SKILL.md': '读取该阶段的任务说明与证据接口。',
            'read_evidence.py': '按版块或样本ID读取保存的测量证据。',
            'run_experiments.py': '提交匹配条件的实验计划，返回保存的测量结果。',
            'run_tests.py': '根据冻结解释生成独立文本，并由目标模型测量。',
            'submit_interpretation.py': '提交初步claim和响应规则，继续进行固定预测步骤。',
            'complete_stage': '保存阶段内容与测量，并生成交接页面。',
            'submit_revision': '提交修订后的解释、Criticism和图注。',
            'complete.py': '保存当前阶段内容并生成报告。',
        }
        selected.append({'step': call.get('step'), 'name': call.get('name', 'bash'),
                         'arguments': {'command': line}, 'result_summary': summaries[kind]})
    return selected


def export(source, target):
    overview = read(source / 'data/overview.json')
    target.mkdir(parents=True, exist_ok=True)
    content = target / 'content'
    traces = []
    for meta in overview['traces']:
        trace = read(source / 'traces' / (meta['id'] + '.json'))
        files = [{'name': f['name'], 'text': f['text']} for f in trace['skill_files']]
        entry = {k: trace[k] for k in ('id', 'title', 'task', 'systems', 'model_rounds', 'tool_calls')}
        entry.update(skill_files=files, calls=action_examples(trace), plans=[],
                     assistant_messages=[], requests=[], public_excerpt=True)
        if trace.get('submission'):
            name = meta['id'] + '-submission.json'
            write(content / name, read(source / trace['submission']))
            entry['submission'] = 'content/' + name
        for plan in trace.get('plans', []):
            if plan['name'] not in ('round1.json', 'round2.json', 'completion.json'):
                continue
            name = meta['id'] + '-' + plan['name']
            write(content / name, read(source / plan['file']))
            entry['plans'].append({'name': plan['name'], 'file': 'content/' + name})
        write(content / (meta['id'] + '.json'), entry)
        traces.append({k: entry[k] for k in ('id', 'title', 'model_rounds', 'tool_calls', 'plans')})

    snapshots = []
    for saved in overview['snapshots']:
        original = source / saved['file']
        html = report(original.read_text())
        name = original.name
        compressed = gzip.compress(html.encode(), mtime=0)
        path = target / 'reports/payloads' / (name + '.gz')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(compressed)
        (target / 'reports' / name).write_text(loader(name + '.gz'))
        snapshots.append({k: saved[k] for k in ('id', 'title', 'claim', 'fields')})
        snapshots[-1]['file'] = 'reports/' + name

    auxiliary = []
    selectors = {
        'initial-prediction': '/01-interpretation/response-predictor',
        'experiment-prediction': '/02-experiments/response-predictor',
        'reader': '/03-revision/paper-reader',
        'final-prediction': '/03-revision/response-predictor',
        'attempt-labeler': '/02-experiments/spectrum-labeler',
        'blind-writer': '/02-experiments/blind-writer',
        'refresh': '/refresh/response-predictor',
    }
    for kind, pattern in selectors.items():
        item = next((x for x in overview['aux_calls'] if pattern in x['route'] and
                     (kind != 'blind-writer' or x['id'].startswith('measurement-source-'))), None)
        if item is None:
            continue
        name = 'prompt-' + kind + '.json'
        example = prompt_example(source, item)
        write(content / name, example)
        auxiliary.append({'id': kind, 'kind': kind, 'file': 'content/' + name})

    stages = public(overview['stages'])
    for name, stage in stages.items():
        old = overview['stages'][name]
        output = 'content/' + name + '-interpretation.json'
        write(target / output, read(source / old['output']))
        stage['output'] = output
        for pred in stage['predictions']:
            pred['examples'] = pred.get('examples', [])[:2]
            pred.pop('output', None)
    result = {k: public(overview[k]) for k in ('unit', 'model', 'preparation',
               'initial_files', 'reuse_plan', 'refresh', 'measurements')}
    result.update(traces=traces, snapshots=snapshots, aux_calls=auxiliary, stages=stages)
    write(content / 'episode.json', result)
    write(content / 'demo-evidence.json', read(source / 'data/demo-evidence.json'))
    write(content / 'preparation.json', public(overview['preparation']))
    write(content / 'measurements.json', public(overview['measurements']))
    print('Exported', len(snapshots), 'reports and', len(auxiliary), 'prompt examples to', target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Local recorded demo directory')
    parser.add_argument('--output', type=Path, default=ROOT / 'website')
    args = parser.parse_args()
    export(args.source.resolve(), args.output.resolve())
