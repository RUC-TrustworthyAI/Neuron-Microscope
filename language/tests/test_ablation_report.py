"""Exercise the ablation report script with saved-format bundles and no model calls."""
import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

from language.tools.report import replace_ablation, source_bundle


TEMPLATE = Path(__file__).resolve().parents[1] / 'analysis/report.html'
DOM = """
class Element {
  constructor(tag, text = '') {
    this.tag = tag;
    this.nodeType = 1;
    this.text = text;
    this.attrs = {};
    this.style = {};
    this.children = [];
  }
  setAttribute(key, value) { this.attrs[key] = value; }
  append(child) { child.parent = this; this.children.push(child); }
  remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
}
const document = {
  createElement: tag => new Element(tag),
  createTextNode: text => new Element('#text', text),
};
let page;
const sec = (_, title, intro, figures) => { page = $('section', null, title, intro, figures); };
const fig = (node, caption) => $('figure', null, node, $('figcaption', null, caption));
const P = (...children) => $('p', null, ...children);
const descendants = node => [node, ...node.children.flatMap(descendants)];
const text = node => descendants(node).map(item => item.text).join('');
const table = label => descendants(page).find(node => node.attrs['aria-label'] === label);
const rows = node => node.children.find(child => child.tag === 'tbody').children;
"""


def evaluate(bundle, code):
    source = TEMPLATE.read_text(encoding='utf-8')
    helpers = source[source.index('const POS ='):source.index('// Per-token shading:')]
    effect = source[source.index('// 5b. What it does:'):source.index('// 5c. Where it sits:')]
    script = 'const B = ' + json.dumps(bundle) + ';\n' + DOM + helpers + effect + '\n' + code
    result = subprocess.run(['node', '-'], input=script, text=True, encoding='utf-8',
                            env={**os.environ, 'LANG': 'en_US.UTF-8'}, capture_output=True)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


def ablation_row(**fields):
    return {
        'kind': 'controlled', 'text': 'long context ' * 20 + 'scored anchor',
        'peak_index': 24, 'peak_token': ' anchor', 'kl': .016, 'kl_controls': [.001, .002, .003],
        'top_base': [{'token': ' A', 'token_id': 11, 'p': .8}],
        'top_ablated': [{'token': ' B', 'token_id': 12, 'p': .6}],
        'gains': [{'token': ' B', 'dp': .4}],
        'losses': [{'token': token, 'dp': -.05} for token in (' A', ' C', ' D', ' E')],
        **fields,
    }


class AblationReportReplayTests(unittest.TestCase):
    def test_replay_replaces_the_saved_effect_renderer_and_retains_other_sections(self):
        saved = {'neuron': {'id': 'L6_N1169'}, 'claim': {'title': 'Saved claim'},
                 'ablation': {'rows': [{'kind': 'graph'}]},
                 'section_prose': {'ablation': {'summary': 'Old graph-only evidence'},
                                   'anatomy': {'summary': 'Saved anatomy'}},
                 'criticism': [{'round': 1, 'attack': 'Saved criticism'},
                               {'round': 2, 'attack': 'Saved follow-up'}],
                 'followup_status': {'status': 'budget_exhausted'}}
        html = f'''<style>@media (max-width:1000px){{.topnav a[href='#family'],.topnav a[href='#criticism']{{display:none}}}}</style>
<script id="bundle" type="application/json">{json.dumps(saved)}</script>
<script>
const B = JSON.parse(document.getElementById('bundle').textContent);
const POS = '#d9480f';
// 5b. What it does: old rendering
throw new Error('old Effect renderer');
// 5c. Where it sits: preserved rendering
</script>'''
        section = {'rows': [ablation_row(text='Causal prefix </script>')], 'summary': {'n': 1}}
        page, bundle = replace_ablation(html, section)
        self.assertEqual(source_bundle(page)[0], bundle)
        self.assertEqual(bundle['ablation'], section)
        self.assertEqual(bundle['section_prose'], {'anatomy': saved['section_prose']['anatomy']})
        for field in ('neuron', 'claim', 'criticism', 'followup_status'):
            self.assertEqual(bundle[field], saved[field])
        self.assertIn(".topnav a[href='#family']{display:none}", page)
        self.assertNotIn(".topnav a[href='#criticism']{display:none}", page)
        self.assertIn('Coverage by cohort', page)
        self.assertIn('// 5c. Where it sits: preserved rendering', page)
        self.assertNotIn('old Effect renderer', page)
        self.assertNotIn('Causal prefix </script>', page)


@unittest.skipUnless(shutil.which('node'), 'Node is required to execute the report script')
class AblationReportTests(unittest.TestCase):
    def test_coverage_and_before_after_keep_shortfalls_and_the_scored_token_visible(self):
        observed = ablation_row(kind='strong-positive', case_id='corpus-123', next_token=' observed',
                                next_logprob_base=-2.302585092994046, next_logprob_ablated=-4.605170185988092,
                                baseline_top_logprob_ablated=-1.6094379124341003,
                                control_neurons=[9, 4, 5], group='Fixture group', saved_product=1.75,
                                live_product=2., relative_product_magnitude=.75)
        predicted = ablation_row(kl=.009, baseline_top_logprob_ablated=-1.6094379124341003,
                                 condition_label='Fixture condition')
        bundle = {'neuron': {'layer': 6}, 'ablation': {'rows': [predicted, observed], 'summary': {
            'n': 2, 'expected': 5, 'controls': [9, 4, 5],
            'coverage': {'strong-positive': {'requested': 3, 'selected': 2, 'completed': 1,
                                             'median_kl': .016, 'median_kl_controls': .002},
                         'controlled': {'requested': 2, 'selected': 1, 'completed': 1,
                                        'median_kl': .009, 'median_kl_controls': .002}},
        }}}
        actual = evaluate(bundle, """
const [observed, predicted] = rows(table('Per-example ablations'));
console.log(JSON.stringify({
  coverage: rows(table('Coverage by cohort')).map(row => row.children.map(text)),
  comparison: [observed, predicted].map(row => text(row.children[3])),
  details: text(observed.children[0]), condition: text(predicted.children[0]),
  losses: text(observed.children[4]),
  page: text(page),
}));
""")
        self.assertEqual(actual['coverage'][0], ['Positive tail', '3', '2', '1', '0.016', '0.002'])
        self.assertEqual(actual['coverage'][1], ['Controlled', '2', '1', '1', '0.009', '0.002'])
        self.assertEqual(actual['comparison'], ['" observed" 10% → 1%Observed next token',
                                                '" A" 80% → 20%Baseline top prediction'])
        self.assertIn('2 / 5', actual['page'])
        self.assertIn('Ablation at the scored token', actual['page'])
        self.assertIn(observed['text'], actual['details'])
        self.assertIn('Case ID corpus-123', actual['details'])
        self.assertIn('Saved activation (product) 1.75', actual['details'])
        self.assertIn('Measured activation (product) 2', actual['details'])
        self.assertIn('Saved magnitude relative to same-sign extreme 75%', actual['details'])
        self.assertIn('Fixture group', actual['details'])
        self.assertIn('Fixture condition', actual['condition'])
        self.assertIn('Top predictions after ablation " B" 60%', actual['details'])
        self.assertIn('Probability gains " B" 0.4', actual['details'])
        self.assertIn('" E" -0.05', actual['details'])
        self.assertNotIn('" E"', actual['losses'])
        for control in ['L6/N9 · 0.001', 'L6/N4 · 0.002', 'L6/N5 · 0.003']:
            self.assertIn(control, actual['details'])

    def test_controlled_next_token_precedes_the_baseline_top_prediction(self):
        controlled = ablation_row(next_token=' fixed', next_logprob_base=-2.302585092994046,
                                  next_logprob_ablated=-4.605170185988092,
                                  baseline_top_logprob_ablated=-1.6094379124341003)
        legacy = {**controlled, 'kind': 'ladder:legacy'}
        bundle = {'ablation': {'rows': [controlled, legacy], 'summary': {'n': 2}}}
        comparison = evaluate(bundle, """
console.log(JSON.stringify(rows(table('Per-example ablations')).map(row => text(row.children[3]))));
""")
        self.assertEqual(comparison, ['" fixed" 10% → 1%Controlled next token',
                                      '" fixed" 10% → 1%Observed next token'])

    def test_saved_bundle_can_show_all_rows_and_keeps_missing_probability_unavailable(self):
        old_rows = [ablation_row(kl=i / 100, kind='ladder:test', text=f'Example {i}') for i in range(13)]
        old_rows[0]['top_ablated'].append({'token': ' A', 'token_id': 11, 'p': .3})
        bundle = {'ablation': {'rows': old_rows, 'summary': {'n': 13, 'controls': [31, 32, 33]}}}
        actual = evaluate(bundle, """
const results = table('Per-example ablations');
const more = descendants(page).find(node => node.tag === 'button');
const initial = rows(results);
const before = {n: initial.length, top: text(initial[0].children[0]), button: text(more),
                unknown: text(initial[0].children[3]), control: text(initial[0].children[0])};
more.onclick();
const all = rows(results);
console.log(JSON.stringify({before, n: all.length, last: text(all.at(-1).children[3]),
                            expanded: !descendants(page).includes(more), coverage: !!table('Coverage by cohort')}));
""")
        self.assertEqual(actual['before']['n'], 12)
        self.assertIn('Example 12', actual['before']['top'])
        self.assertEqual(actual['before']['button'], 'Show 1 more')
        self.assertIn('80% → –', actual['before']['unknown'])
        self.assertIn('N31 · 0.001', actual['before']['control'])
        self.assertEqual(actual['n'], 13)
        self.assertIn('80% → 30%', actual['last'])
        self.assertTrue(actual['expanded'])
        self.assertFalse(actual['coverage'])

    def test_empty_measurements_keep_the_coverage_summary(self):
        bundle = {'ablation': {'rows': [], 'summary': {'n': 0, 'expected': 4, 'coverage': {
            'weak': {'requested': 4, 'selected': 0, 'completed': 0},
        }}}}
        actual = evaluate(bundle, """
console.log(JSON.stringify({page: text(page), coverage: rows(table('Coverage by cohort')).map(text),
                            examples: !!table('Per-example ablations')}));
""")
        self.assertIn('0 / 4', actual['page'])
        self.assertIn('No ablations completed.', actual['page'])
        self.assertEqual(actual['coverage'], ['Weak400––'])
        self.assertFalse(actual['examples'])


if __name__ == '__main__':
    unittest.main()
