import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from language.workflow.anatomy_replay import replay


class AnatomyReplayTests(unittest.TestCase):
    def test_replay_adds_pair_to_completed_report_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'investigation'
            output = root / 'anatomy'
            (source / 'report').mkdir(parents=True)
            bundle = {'neuron': {'id': 'L1_N2'}, 'anatomy': {'strip': {'tokens': ['x']}},
                      'display_selection': {'headline_figure_id': 'sweep'},
                      'ablation': {'rows': []}, 'criticism': [{'round': 1}]}
            page = source / 'report/index.html'
            page.write_text(
                "<style>@media (max-width:1000px){.topnav a[href='#anatomy'],"
                ".topnav a[href='#family']{display:none}}</style>\n"
                f'<script id="bundle">{json.dumps(bundle)}</script>\n'
                "<script>const B = JSON.parse(document.getElementById('bundle').textContent);\n"
                "const POS = '#d9480f';\n"
                "  if (has(A.headline)) figs.push(legacy);\n"
                "  sec({ id: 'anatomy', wide: true }, 'Inside the neuron', [P('Old intro')], figs);\n"
                '</script>', encoding='utf-8')
            (source / 'status.json').write_text('{"phase":"complete"}', encoding='utf-8')
            (source / 'config.json').write_text('{"neuron":"L1_N2"}', encoding='utf-8')
            (source / 'artifacts.json').write_text('{"figures":[],"all_measured":[]}', encoding='utf-8')
            before = page.read_bytes()
            pair = {'figure_id': 'sweep', 'rows': [{'case_id': 'a'}, {'case_id': 'b'}]}
            with patch('language.workflow.anatomy_replay.select_anatomy_pair', return_value=pair) as select:
                report = replay(source, output)
                select.assert_called_once_with([], [], preferred_figure_id='sweep')
                self.assertEqual(replay(source, output), report)
            new_bundle = json.loads((report / 'paper-data.json').read_bytes())['bundle']
            self.assertEqual(new_bundle.pop('anatomy'), {**bundle.pop('anatomy'), 'pair': pair})
            self.assertEqual(new_bundle, bundle)
            self.assertEqual(page.read_bytes(), before)
            self.assertEqual(json.loads((output / 'anatomy-pair.json').read_bytes()), pair)
            rendered = (report / 'index.html').read_text(encoding='utf-8')
            self.assertIn('if (!hasAnatomyPair && has(A.headline))', rendered)
            self.assertIn('SiLU preserves the sign of the raw gate', rendered)
            self.assertNotIn('Old intro', rendered)
            self.assertNotIn(".topnav a[href='#anatomy']", rendered)
            self.assertEqual(json.loads((output / 'status.json').read_bytes())['phase'], 'complete')
            marker = output / 'anatomy-source.json'
            receipt = json.loads(marker.read_bytes())
            marker.write_text(json.dumps({**receipt, 'files': {'report/index.html': 'legacy digest'}}))
            page.write_bytes(before + b'\n<!-- updated source report -->')
            with patch('language.workflow.anatomy_replay.select_anatomy_pair', return_value=pair):
                replay(source, output)
                self.assertIn('updated source report', (report / 'index.html').read_text())
                self.assertEqual(json.loads(marker.read_bytes()), receipt)
                marker.write_text(json.dumps({**receipt, 'neuron': 'L2_N3'}))
                with self.assertRaisesRegex(ValueError, 'another investigation'):
                    replay(source, output)
            with self.assertRaisesRegex(ValueError, 'separate output directory'):
                replay(source, source / 'nested')
