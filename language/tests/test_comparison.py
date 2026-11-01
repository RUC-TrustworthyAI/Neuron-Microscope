"""Publication controls must never modify the scientific handoff or its reference."""
from pathlib import Path
import json
import re
import tempfile
import unittest
from comparison import publish, blind_score


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
        from report_directory import render
        page = render([{'neuron': 'L24_N17907', 'blind_score': result}])
        self.assertIn('>0.991</span>', page)


if __name__=='__main__':unittest.main()
