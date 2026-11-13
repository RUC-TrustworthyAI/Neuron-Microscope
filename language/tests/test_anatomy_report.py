"""Exercise the Anatomy report's measured pair and saved-report fallback."""
import json
import shutil
import subprocess
import unittest

from language.tests.test_ablation_report import DOM, TEMPLATE


def evaluate_anatomy(anatomy, code):
    source = TEMPLATE.read_text(encoding='utf-8')
    helpers = source[source.index('const POS ='):source.index('// Per-token shading:')]
    anatomy_script = source[source.index('// 5a. Inside the neuron:'):source.index('// 5b. What it does:')]
    script = 'const B = ' + json.dumps({'anatomy': anatomy}) + ';\n' + DOM + helpers + anatomy_script + '\n' + code
    result = subprocess.run(['node', '-'], input=script, text=True, encoding='utf-8', capture_output=True)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


def anatomy_row(case_id, label, tokens, gate, up, act, anchor):
    return {'case_id': case_id, 'label': label, 'text': ''.join(tokens), 'tokens': tokens,
            'token_ids': list(range(50, 50 + len(tokens))), 'gate': gate, 'up': up, 'act': act,
            'replacement_positions': list(range(1, anchor)), 'scored_positions': [anchor],
            'anchor_index': anchor}


@unittest.skipUnless(shutil.which('node'), 'Node is required to execute the report script')
class AnatomyReportTests(unittest.TestCase):
    def test_paired_conditions_share_channel_scales_and_mark_saved_positions(self):
        low = anatomy_row('sweep:0', 'Brief', ['Keep', ' brief', ','], [1., 0., 2.],
                          [0., -.1, 2.], [0., 0., 3.5234375], 2)
        high = anatomy_row('sweep:1', 'Longer', ['Keep', ' much', ' longer', ','], [1., 0., 0., 4.],
                           [0., 0., 0., 2.], [0., 0., 0., 7.84375], 3)
        anatomy = {'pair': {'figure_id': 'sweep', 'title': 'Closing comma', 'template': 'Keep {slot},',
                            'comparison': 'Same scored comma', 'explanation': 'Raw gate increases at the comma.',
                            'rows': [low, high]},
                   'headline': [{'tokens': ['Legacy trace should be hidden'], 'gate': [1], 'up': [1], 'act': [.73]}]}
        actual = evaluate_anatomy(anatomy, """
const sections = descendants(page);
const pair = sections.find(node => node.attrs.id === 'anatomy-comparison');
const conditions = descendants(pair).filter(node => node.attrs['data-case-id']);
const readCondition = condition => {
  const columns = descendants(condition).filter(node => node.attrs['data-token-index'] != null);
  const anchor = columns.find(column => column.attrs['data-anchor']);
  return {id: condition.attrs['data-case-id'], n: columns.length,
    replaced: columns.filter(column => column.attrs['data-replacement']).map(column => column.attrs['data-token-index']),
    scored: columns.filter(column => column.attrs['data-scored']).map(column => column.attrs['data-token-index']),
    title: anchor.attrs.title, bars: anchor.children.slice(0, 3).map(bar => bar.children[0].style.cssText)};
};
console.log(JSON.stringify({trace: conditions.map(readCondition), page: text(page)}));
""")
        left, right = actual['trace']
        self.assertEqual((left['id'], right['id']), ('sweep:0', 'sweep:1'))
        self.assertEqual((left['n'], right['n']), (3, 4))
        self.assertEqual((left['replaced'], right['replaced']), ([1], [1, 2]))
        self.assertEqual((left['scored'], right['scored']), ([2], [3]))
        self.assertIn('Token 2 · "," · ID 52', left['title'])
        self.assertIn('Product 3.5234375', left['title'])
        self.assertIn('Selected anchor', left['title'])
        self.assertTrue(left['bars'][0].startswith('height:11px;'))
        self.assertTrue(right['bars'][0].startswith('height:22px;'))
        self.assertEqual(left['bars'][1], right['bars'][1])
        self.assertNotEqual(left['bars'][2], right['bars'][2])
        self.assertIn('Raw gate increases at the comma.', actual['page'])
        self.assertIn('negative when the two inputs have opposite signs', actual['page'])
        self.assertNotIn('Legacy trace should be hidden', actual['page'])

    def test_saved_headline_renders_when_pair_is_absent(self):
        legacy = {'headline': [{'tokens': ['Historical condition'], 'gate': [0.], 'up': [0.], 'act': [0.]}]}
        page = evaluate_anatomy(legacy, 'console.log(JSON.stringify(text(page)));')
        self.assertIn('Historical condition', page)
        self.assertNotIn('Paired conditions', page)


if __name__ == '__main__':
    unittest.main()
