"""Optional renderer integration test using an activation-only report fixture."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from language.workflow.state import write


class ReportSynthesisTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node is needed to check report links')
    def test_only_claim_links_to_evidence_and_opens_samples(self):
        renderer = Path(__file__).resolve().parents[1] / 'tools/report_synthesis.js'
        fixture = r"""
const assert = require('node:assert/strict');
class Element {
  constructor(tag, text = '') {
    this.tag = tag;
    this.text = text;
    this.attrs = {};
    this.children = [];
  }
  append(child) { this.children.push(child); }
  replaceChildren(...children) { this.children = children; }
  remove() {}
}
const document = {};
const B = {};
const callbacks = {};
const window = {addEventListener: (type, handler) => { callbacks[type] = handler; }};
const location = {hash: '#evidence-criticism%3Aexample'};
const $ = (tag, attrs, ...children) => {
  const node = new Element(tag);
  node.attrs = {...attrs};
  node.addEventListener = () => {};
  for (const child of children.flat(Infinity)) node.append(child instanceof Element ? child : new Element('#text', child));
  return node;
};
const P = (...children) => $('p', null, ...children);
const descendants = node => [node, ...node.children.flatMap(descendants)];
const text = node => descendants(node).map(item => item.text).join('');
const before = new Element('div');
const claimFigures = $('div', null, before);
claimFigures.querySelector = () => before;
before.replaceWith = node => claimFigures.append(node);
const headline = new Element('h1');
const lede = new Element('p');
const sample = new Element('div');
sample.querySelector = () => sample.children.find(node => node.attrs.class === 'cap scored-evidence');
const folded = {tagName: 'DETAILS'};
sample.parentElement = folded;
let focused;
sample.focus = () => { focused = sample; };
sample.scrollIntoView = () => {};
const why = new Element('div');
const criticism = {
  querySelector: selector => selector === '.text' ? {querySelector: () => why} : null,
  querySelectorAll: () => [{querySelector: () => ({children: [sample]})}],
};
const data = {
  claim: {title: 'Measured', lede: 'Summary', findings: {pattern: {text: 'Found', evidence_ids: ['criticism:example']}}},
  section_prose: {criticism: {summary: 'Saved criticism summary', evidence_ids: ['criticism:section-only']}},
  evidence: {
    'criticism:example': {section: 'criticism', label: 'Case', locator: {row: 0, sample: 0},
      details: {position: 4, token: ' et', location: 'replacement_span', reduction: 'absmax'}},
    'criticism:section-only': {section: 'criticism', label: 'Old section link'},
  },
};
Object.assign(document, {
  getElementById(id) { return {
    'report-synthesis': {textContent: JSON.stringify(data)},
    'criticism': criticism,
  }[id]; },
  querySelector: selector => ({'#head h1': headline, '#head .lede': lede, '#claim .figs': claimFigures})[selector],
  querySelectorAll: () => [],
});
"""
        checks = r"""
assert.equal(folded.open, true);
assert.equal(focused, sample);
assert.match(text(sample), /Scored token " et" · position 4 \(0-based\) · replacement span/);
assert.equal(text(why), 'Saved criticism summary');
assert.equal(descendants(why).some(node => node.tag === 'a'), false);
const link = descendants(claimFigures).find(node => node.tag === 'a');
assert.equal(link.attrs.href, '#evidence-criticism%3Aexample');
focused = null;
folded.open = false;
location.hash = '';
callbacks.hashchange();
location.hash = link.attrs.href;
callbacks.hashchange();
assert.equal(folded.open, true);
assert.equal(focused, sample);
"""
        script = fixture + renderer.read_text(encoding='utf-8') + checks
        result = subprocess.run(['node', '-'], input=script, encoding='utf-8', capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class WalkthroughExportTest(unittest.TestCase):
    def test_export_preserves_recorded_stages(self):
        import contextlib
        import gzip
        import hashlib
        import io
        from language.examples.export_walkthrough import export, export_archive, report_script
        from language.tools.report import BUNDLE, encode_bundle
        from language.workflow.state import read
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            run, report, output = base / 'run', base / 'source', base / 'export'
            identity = {'id': 'L2_N5', 'layer': 2, 'neuron': 5, 'model': 'Qwen3-0.6B fixture'}
            write(run / 'config.json', {'neuron': 'L2_N5', 'model': str(base / 'weights'),
                                      'report': str(report), 'activations': str(base / 'corpus')})
            write(report / 'report.json', {'identity': identity})
            html = '<html><script id="bundle">{"display_selection":{"headline_figure_id":"control"}}</script></html>'
            (report / 'index.html').write_text(html)
            bundle = {'display_selection': {'headline_figure_id': 'control'},
                      'tests': {'route_prediction': {'summary': {'cases': 1}}},
                      'criticism': [{'attack': 'Original criticism', 'verdict': 'narrowed'},
                                    {'round': 1, 'investigation_id': 'R1_Q1', 'criticism_ids': ['C1'],
                                     'attack': 'Follow-up round 1: Does context distinguish the explanations?',
                                     'result': 'Only the contextual condition activated.', 'verdict': 'survived'}],
                      'ablation': {'rows': [{'kl': .016, 'kl_controls': [.001, .002, .003]}],
                                   'summary': {'n': 1, 'controls': [9, 4, 8]}},
                      'anatomy': {'pair': {'figure_id': 'control', 'rows': [
                          {'case_id': 'control:0', 'gate': [1.], 'up': [2.], 'act': [1.462]},
                          {'case_id': 'control:1', 'gate': [-1.], 'up': [2.], 'act': [-.538]}]}}}
            write(run / 'report/paper-data.json', {'bundle': bundle})
            labels = {'scope': {'archive': '/srv/private/corpus', 'archive_model': '/opt/private/weights',
                                'measurement': {'model': {'path': r'C:\private\served',
                                                          'metadata_before_load': {'path': '/home/user/base'}}},
                                'seed': 42},
                      'labels': {'L2_N5': {'label': 'Context'}}}
            public_labels = {**labels, 'scope': {'archive': 'corpus', 'archive_model': 'weights',
                            'measurement': {'model': {'path': 'served', 'metadata_before_load': {'path': 'base'}}},
                            'seed': 42}}
            synthesis = {'claim': {'title': 'Fixture', 'findings': {'pattern': {'evidence_ids': ['C1']}}},
                         'section_prose': {'criticism': {'summary': 'Saved criticism summary', 'evidence_ids': ['C2']}},
                         'evidence': {'C1': {'label': 'Claim case'}, 'C2': {'label': 'Old section case'}}}
            claim_synthesis = {**synthesis, 'section_prose': {'criticism': {'summary': 'Saved criticism summary'}},
                               'evidence': {'C1': synthesis['evidence']['C1']}}
            legacy_html = (
                f'<html><script id="bundle">{encode_bundle(bundle)}</script>'
                f'<script id="neuron-labels">{json.dumps(labels)}</script>'
                f'<script id="report-synthesis" type="application/json">{json.dumps(synthesis)}</script>'
                '<script id="report-synthesis-render">/* old section citations */</script></html>')
            (run / 'report/index.html').write_text(legacy_html, encoding='utf-8')
            write(run / 'artifacts.json', {'all_measured': [], 'figures': [{'id': 'control', 'results': []}],
                'corpus': {'min': -1., 'max': 1.}, 'tests': {}, 'atlas': [], 'graphs': [],
                'review': {}, 'revision': {'criticism': []}})
            write(run / 'interpretation-history.json', [{'round': 0, 'selected': True,
                'explanation_path': 'explanation.json', 'predictions_path': 'predictions.json', 'summary': {}}])
            write(run / 'explanation.json', {'output': {'title': 'Fixture'}})
            write(run / 'predictions.json', {'records': [], 'response_rules': [], 'summary': {}})
            write(run / 'sample/sections.json', {'spectrum': {'points': [], 'edges': [], 'corpus_hist': {'counts': []}}})
            write(run / 'sample/sampling.json', {'tokens_scanned': 1024, 'sampled_points': 0,
                'population_bin_counts': [], 'sample_bin_counts': [], 'scan_seconds': 0., 'wall_seconds': 0.,
                'new_model_forwards': 0, 'gpu_seconds': 0., 'llm_requests': 0})
            write(run / 'evidence.json', {'statistics': {}, 'samples': [], 'documents': []})
            write(run / 'search-progress.json', [])
            write(run / 'measurements/figures-initial.json', [])
            write(run / 'measurements/figures-revised.json', [])
            followup = {'rounds': [{'round': 1, 'investigations': [{'assessment': {'verdict': 'survived'}}]}]}
            write(run / 'followup/investigation.json', followup)
            for directory in ('api/round-1/assessment', 'followup/api/round-1/assessment',
                              'api/report-synthesis', 'followup/api/report-synthesis'):
                call = run / directory
                write(call / 'result.json', {'output': {'verdict': 'survived'}})
                write(call / 'attempt-1/request.json', {'model': 'Fixture', 'messages': [
                    {'content': 'Assess the evidence.'}, {'content': '{}'}]})
            call = run / 'followup/api/report-synthesis'
            messages = [{'role': 'system', 'content': 'Synthesize.'}, {'role': 'user', 'content': '{}'}]
            correction = [{'role': 'assistant', 'content': 'Invalid response.'},
                          {'role': 'user', 'content': 'Return JSON.'}]
            fingerprint = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
            write(call / 'result.json', {'input_id': fingerprint, 'output': {'verdict': 'survived'}})
            write(call / 'attempt-2/request.json', {'model': 'Fixture', 'messages': messages + correction})
            write(call / 'attempt-2/completion.json', {'finish_reason': 'stop'})
            write(call / 'attempt-3/request.json', {'model': 'Fixture', 'messages': messages})
            for directory in ('api/report-synthesis/overview/title', 'followup/api/report-synthesis/sections/criticism'):
                part = run / directory
                write(part / 'result.json', {'attempt': 'attempt-1', 'output': {'text': 'Measured result.'}})
                write(part / 'attempt-1/request.json',
                      {'model': 'Fixture', 'input': messages, 'text': {'format': {'type': 'text'}}})
            with contextlib.redirect_stdout(io.StringIO()):
                export(run, output)
            episode = read(output / 'content/episode.json')
            self.assertEqual(episode['subject_model'], identity['model'])
            self.assertEqual(episode['source_identity'], identity)
            self.assertEqual(episode['dataset'], 'Saved native activation corpus')
            self.assertEqual(episode['route_prediction'], bundle['tests']['route_prediction'])
            self.assertEqual(episode['neuron_labels'], public_labels)
            self.assertEqual(episode['followup']['rounds'], followup['rounds'])
            self.assertEqual(episode['synthesis'], claim_synthesis)
            calls = {entry['id']: entry['file'] for entry in episode['calls']}
            self.assertEqual(calls, {'round-1/assessment': 'content/calls/round-1--assessment.json',
                                    'followup/round-1/assessment': 'content/calls/followup/round-1--assessment.json',
                                    'followup/report-synthesis': 'content/calls/followup/report-synthesis.json',
                                    'followup/report-synthesis/sections/criticism':
                                        'content/calls/followup/report-synthesis--sections--criticism.json'})
            section_call = read(output / calls['followup/report-synthesis/sections/criticism'])
            self.assertEqual(section_call['response_format'], 'text')
            self.assertEqual(section_call['output'], {'text': 'Measured result.'})
            final_gzip = output / 'reports/final.html.gz'
            renderer = Path(__file__).resolve().parents[1] / 'tools/report_synthesis.js'
            html = gzip.decompress(final_gzip.read_bytes()).decode()
            self.assertEqual(report_script(html, 'report-synthesis'), claim_synthesis)
            self.assertIn(renderer.read_text(encoding='utf-8'), html)
            # An archived walkthrough may still contain the old section links.
            final_gzip.write_bytes(gzip.compress(legacy_html.encode('utf-8'), mtime=0))
            write(output / 'content/episode.json', {**episode, 'synthesis': synthesis})
            export_archive(output, output)
            self.assertEqual(read(output / 'content/episode.json'), episode)
            self.assertEqual(read(output / calls['followup/round-1/assessment'])['output'], {'verdict': 'survived'})
            self.assertEqual(read(output / calls['followup/report-synthesis'])['correction_messages'], correction)
            exported_html = gzip.decompress(final_gzip.read_bytes()).decode()
            self.assertEqual(report_script(exported_html, 'report-synthesis'), claim_synthesis)
            self.assertIn(renderer.read_text(encoding='utf-8'), exported_html)
            exported_bundle = json.loads(BUNDLE.search(exported_html)[2])
            self.assertEqual(report_script(exported_html, 'neuron-labels'), public_labels)
            for field in ('criticism', 'ablation', 'anatomy'):
                self.assertEqual(exported_bundle[field], bundle[field])

            for field, text in {'title': 'New title', 'lede': 'New lede'}.items():
                part = call / 'overview' / field
                write(part / 'result.json', {'attempt': 'attempt-1', 'output': {'text': text}})
                write(part / 'attempt-1/request.json',
                      {'model': 'Fixture', 'input': messages, 'text': {'format': {'type': 'text'}}})
            with contextlib.redirect_stdout(io.StringIO()):
                export(run, output)
            self.assertIn('followup/report-synthesis',
                          {entry['id'] for entry in read(output / 'content/episode.json')['calls']})
            synthesis['claim'].update(title='New title', lede='New lede')
            (run / 'report/index.html').write_text(
                f'<script id="bundle">{encode_bundle(bundle)}</script>'
                f'<script id="report-synthesis">{json.dumps(synthesis)}</script>')
            with contextlib.redirect_stdout(io.StringIO()):
                export(run, output)
            call_ids = {entry['id'] for entry in read(output / 'content/episode.json')['calls']}
            self.assertNotIn('followup/report-synthesis', call_ids)
            self.assertIn('followup/report-synthesis/overview/lede', call_ids)
            self.assertTrue((call / 'result.json').is_file())

            bundle['criticism'] = bundle['criticism'][:1]
            (run / 'report/index.html').write_text(
                f'<script id="bundle">{encode_bundle(bundle)}</script>'
                f'<script id="report-synthesis">{json.dumps(synthesis)}</script>')
            write(run / 'status.json', {'phase': 'complete'})
            with contextlib.redirect_stdout(io.StringIO()):
                export(run, output)
            episode = read(output / 'content/episode.json')
            self.assertNotIn('followup', episode)
            self.assertEqual({entry['id'] for entry in episode['calls']},
                             {'round-1/assessment', 'report-synthesis', 'report-synthesis/overview/title'})

    def test_archive_rejects_escaped_calls_before_writes(self):
        from language.examples.export_walkthrough import export_archive

        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            archive, output = base / 'archive', base / 'export'
            valid = 'content/calls/first.json'
            write(archive / valid, {})
            outside = base / 'outside.json'
            write(outside, {})
            for path in ('../outside.json', 'content/calls/../../../outside.json', str(outside)):
                with self.subTest(path=path):
                    write(archive / 'content/episode.json', {'calls': [{'file': valid}, {'file': path}]})
                    with self.assertRaisesRegex(ValueError, 'Invalid archive path'):
                        export_archive(archive, output)
                    self.assertFalse(output.exists())


@unittest.skipUnless(os.environ.get('WORKFLOW_TEST_REPORT'), 'Set an activation-only report fixture')
class RendererTest(unittest.TestCase):
    def test_interpretation_preview_shares_the_same_blind_labels(self):
        from language.workflow.render import assemble_interpretation
        from spectrum import build
        report = Path(os.environ['WORKFLOW_TEST_REPORT'])
        data = json.loads((report / 'report.json').read_text())
        sample = build(data, [], {})
        sample['points'] = sample['points'][:2]
        for point in sample['points']:
            gate, up, activation = data['points'][point['id']]['values']
            point.update(gate=gate, up=up, activation=activation)
        rule = dict(id='n', name='Negative hypothesis', description='Fixture', predicted_pole='negative')
        predictions = dict(version='fixture', response_rules=[rule], records=[
            dict(id=p['id'], group='n', predicted_pole='negative', reason='Fixture') for p in sample['points']])
        with tempfile.TemporaryDirectory() as folder:
            output = assemble_interpretation(Path(folder), report, {'claim': {'title': 'Fixture'}}, sample, predictions)
            bundle = json.loads((output / 'paper-data.json').read_text())['bundle']
            self.assertEqual(bundle['spectrum']['points'], bundle['anatomy']['points'])
            self.assertTrue(all(p['predicted_pole'] == 'negative' for p in bundle['spectrum']['points']))
            self.assertIsNone(bundle['custom'])
            self.assertIsNone(bundle['graphs_curated'])
            self.assertIn("new Set(['claim', 'spectrum', 'anatomy'])", (output / 'index.html').read_text())

    def test_numeric_fields_labels_and_shared_points(self):
        from language.workflow.evidence import initial_packet, evidence_catalog
        from language.workflow.render import assemble
        from spectrum import build
        report = Path(os.environ['WORKFLOW_TEST_REPORT'])
        data = json.loads((report / 'report.json').read_text())
        packet = initial_packet(report)
        sample = build(data, [], {})
        sample['points'] = sample['points'][:2]
        for point in sample['points']:
            gate, up, activation = data['points'][point['id']]['values']
            point.update(gate=gate, up=up, activation=activation)
        rule = {'id': 'p', 'name': 'Context', 'description': 'Fixture', 'predicted_pole': 'positive'}
        pred = {'version': 'fixture', 'records': [dict(id=x['id'], group='p', predicted_pole='positive',
                reason='Fixture') for x in sample['points']], 'response_rules': [rule]}
        trace = [{'position': 0, 'token': 'Condition', 'product': 1., 'gate': 1., 'value': 1.,
                  'token_id': 1, 'offset': [0, 9]}]
        rows = [{'id': f'control:{i}', 'label': f'Case{i}', 'text': f'Long test text {i}.',
                 'status': 'complete', 'trace': trace, 'selected': trace[0], 'scored_positions': [0]}
                for i in range(10)]
        figure = {'id': 'control', 'kind': 'sweep', 'title': 'Fixture', 'template': '{slot}',
                  'results': [{'tag': x['label'], 'text': x['text'], 'activation': 1., 'filler': x['label']}
                              for x in rows]}
        claim = dict(title='Test fixture', lede='Fixture', one_paragraph='Fixture', negative_pole='Fixture',
                     what_would_change_my_mind='Fixture', confidence='low')
        artifacts = {'initial': {}, 'revision': {'claim': claim, 'section_prose': {}, 'clusters': [],
            'units': [], 'graph_captions': [], 'criticism': []}, 'review': {'group': 'unresolved',
            'organized': 1, 'understood': 1, 'one_line': 'Fixture', 'concerning': [], 'fine': []},
            'logit_effect': packet['logit_effect'], 'figures': [figure], 'search_regions': [],
            'corpus': {'max': 1., 'min': -1.}, 'atlas_sections': {}, 'sample': sample,
            'predictions': pred, 'atlas': [], 'graphs': [], 'all_measured': rows, 'tests': {}}
        graph = {'graph': {'id': 'graph-1', 'text': 'Condition', 'position': 0}, 'trace': trace}
        catalog = evidence_catalog(rows, [], [graph], sample, packet)
        self.assertIn(sample['points'][0]['id'], catalog)
        self.assertEqual(catalog['graph-1']['activation'], 1.)
        artifacts['evidence'] = catalog
        artifacts['revision']['criticism'] = [{'attack': 'Fixture', 'prediction': 'Fixture',
            'result': 'Fixture', 'verdict': 'narrowed', 'evidence_ids': ['graph-1']}]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root / 'timeline.json', [])
            output = assemble(root, report, 'L57_N2065', artifacts, 4)
            bundle = json.loads((output / 'paper-data.json').read_text())['bundle']
            self.assertEqual(len(bundle['custom']['figures'][0]['results']), 10)
            self.assertTrue(bundle['logit_effect']['promote'])
            self.assertTrue(bundle['logit_effect']['suppress'])
            self.assertEqual(bundle['anatomy']['points'], bundle['spectrum']['points'])
            self.assertTrue(bundle['anatomy']['strip']['case_id'])
            self.assertEqual(bundle['criticism'][0]['evidence'][0]['activation'], 1.)
            self.assertEqual([x['tag'] for x in bundle['headline']['comparison']['fillers']],
                             [f'Case{i}' for i in range(10)])


if __name__ == '__main__':
    unittest.main()
