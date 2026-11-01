"""Publication controls must never modify the scientific handoff or its reference."""
from pathlib import Path
import json
import re
import tempfile
import unittest
from comparison import publish, blind_score, search_score
from report_directory import comparison_rows, render


def report_html(bundle):
    return ('<html><head></head><body><script id="bundle" type="application/json">' + json.dumps(bundle) +
            '</script><script>const B = JSON.parse(document.getElementById("bundle").textContent);'
            'const POS = "red";</script></body></html>')


class ComparisonTests(unittest.TestCase):
    def test_reference_is_added_only_to_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'handoff';source.mkdir()
            html=report_html({'tests': {'blind_max': {'achieved': 20}, 'blind_min': {'achieved': -5}}})
            (source/'index.html').write_text(html)
            refs=root/'references';refs.mkdir()
            original=report_html({'tests': {'blind_max': {'achieved': 10}, 'blind_min': {'achieved': -10}}})
            (refs/'L1_N2.html').write_text(original)
            output=root/'published'
            self.assertTrue(publish(source,output,'L1_N2',refs))
            self.assertEqual((source/'index.html').read_text(),html)
            self.assertFalse((source/'comparison').exists())
            self.assertEqual((output/'comparison/original.html').read_text(),original)
            script=re.search(r'src="(comparison/viewer\.[0-9a-f]{12}\.js)"',(output/'index.html').read_text())
            self.assertIsNotNone(script)
            self.assertTrue((output/script[1]).is_file())
            self.assertIn('data-blind-score="1.25"', (output/'index.html').read_text())
            self.assertIn('data-directory="../"', (output/'index.html').read_text())

    def test_no_reference_still_publishes_independent_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'handoff';source.mkdir()
            html='<html><head></head><body>synthetic report</body></html>'
            (source/'index.html').write_text(html)
            output=root/'published'
            self.assertFalse(publish(source,output,'L1_N2',root/'absent'))
            self.assertEqual((source/'index.html').read_text(),html)
            self.assertIn('data-directory="../"', (output/'index.html').read_text())
            self.assertNotIn('data-reference=', (output/'index.html').read_text())
            self.assertFalse((output/'comparison/original.html').exists())

    def test_blind_score_uses_achieved_values_and_retains_large_ratios(self):
        ours={'tests': {'blind_max': {'achieved': 30, 'target': 999}, 'blind_min': {'achieved': -20}}}
        reference={'tests': {'blind_max': {'achieved': 10, 'target': 999}, 'blind_min': {'achieved': -10}}}
        self.assertEqual(blind_score(ours, reference)['score'], 2.5)
        self.assertIsNone(blind_score({}, reference))
        reference['tests']['blind_min']['achieved']=0
        self.assertIsNone(blind_score(ours, reference))

    def test_single_pole_score_preserves_the_missing_measurement(self):
        ours = {'tests': {'blind_max': {'achieved': 53.25}}}
        reference = {'tests': {'blind_max': {'achieved': 53.75}, 'blind_min': {'achieved': -1.78125}}}
        result = blind_score(ours, reference)
        self.assertAlmostEqual(result['score'], 53.25 / 53.75)
        self.assertEqual(result['basis'], 'positive')
        self.assertIsNone(result['negative_ratio'])
        self.assertIsNone(result['reproduced_min'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);source = root / 'report';source.mkdir();refs = root / 'refs';refs.mkdir()
            (source / 'index.html').write_text(report_html(ours))
            (refs / 'L24_N17907.html').write_text(report_html(reference))
            publish(source, root / 'export', 'L24_N17907', refs)
            self.assertIn('data-blind-score-basis="positive"', (root / 'export/index.html').read_text())
        page = render([{'neuron': 'L24_N17907', 'blind_score': result}])
        self.assertIn('>0.991</span>', page)

    def test_search_score_uses_search_extrema(self):
        ours = {'search': {'best_positive': 30, 'best_negative': -20},
                'tests': {'blind_max': {'achieved': 999}}}
        reference = {'search': {'best_positive': 10, 'best_negative': -10}}
        result = search_score(ours, reference)
        self.assertEqual(result['score'], 2.5)
        self.assertEqual(result['positive_ratio'], 3)
        self.assertEqual(result['negative_ratio'], 2)
        self.assertIsNone(search_score({}, reference))
        ours['search']['best_positive'] = None
        self.assertEqual(search_score(ours, reference)['score'], 2)
        reference['search']['best_negative'] = 0
        self.assertIsNone(search_score(ours, reference))

    def test_directory_statistics_use_unrounded_available_scores(self):
        rows = [{'neuron': f'L1_N{i}', 'model': 'Qwen3-32B',
                 'blind_score': {'score': value}, 'search_score': {'score': search}}
                for i, (value, search) in enumerate(zip([0.00049, 0.00049, 0.00149], [1, 3, 9]))]
        rows.append({'neuron': 'L1_N3', 'ready': False})
        page = render(rows, corpus='Reasoning Mix 100M')
        headers = re.search(r'<thead>(.*?)</thead>', page, re.S)[1]
        self.assertIn('Mean 0.001</span>', headers)
        self.assertIn('Median 0.000</span>', headers)
        self.assertIn('Mean 4.333</span>', headers)
        self.assertIn('Median 3.000</span>', headers)
        self.assertEqual(headers.count('of 3 available scores'), 4)
        self.assertEqual(headers.count('<th>'), 5)
        self.assertIn('4 interactive neuron reports · Qwen3-32B · Reasoning Mix 100M', page)
        self.assertIn('L1_N0/?compare=1', page)
        self.assertNotIn('href="L1_N3/', page)
        self.assertNotIn('{{', page)
        self.assertNotIn('Similarity', page)
        self.assertNotIn('Head score', page)
        empty = render([])
        self.assertEqual(empty.count('Mean —</span>'), 2)
        self.assertEqual(empty.count('Median —</span>'), 2)

    def test_directory_reads_both_scores_from_completed_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'L1_N2'
            (folder / 'comparison').mkdir(parents=True)
            ours = {'claim': {'title': 'A <pattern>'}, 'neuron': {'model': 'Qwen3-32B'},
                    'tests': {'blind_max': {'achieved': 20}, 'blind_min': {'achieved': -5}},
                    'search': {'best_positive': 30, 'best_negative': -10}}
            reference = {'claim': {'title': 'Reference & pattern'},
                         'tests': {'blind_max': {'achieved': 10}, 'blind_min': {'achieved': -10}},
                         'search': {'best_positive': 10, 'best_negative': -10}}
            (folder / 'index.html').write_text(report_html(ours))
            (folder / 'comparison/original.html').write_text(report_html(reference))
            rows = comparison_rows(root, ['L1_N2', 'L1_N3'])
            self.assertEqual(rows[0]['blind_score']['score'], 1.25)
            self.assertEqual(rows[0]['search_score']['score'], 2)
            self.assertIsNone(rows[1]['blind_score'])
            self.assertIsNone(rows[1]['search_score'])
            page = render(rows)
            self.assertIn('A &lt;pattern&gt;', page)
            self.assertIn('Reference &amp; pattern', page)
            self.assertIn('Blind min: |-5 / -10|', page)
            self.assertIn('Search max: 30 / 10', page)
            from language.workflow.render import directory
            directory(root, ['L1_N2', 'L1_N3'], None)
            self.assertEqual((root / 'index.html').read_text(), page)


if __name__=='__main__':unittest.main()
