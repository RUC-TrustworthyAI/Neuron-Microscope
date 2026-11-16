"""Offline checkpoint compatibility and in-place report recovery."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from language.tests.followup_fixtures import FakeClient, FakeExecutor, base_packet, plan, question
from language.tools.report import BUNDLE, encode_bundle
from language.workflow.followups import SOURCE_FILES, route_overlay, run
from language.workflow.state import read, write


class FollowupResumeTests(unittest.TestCase):
    def test_resume_preserves_source_and_rejects_changed_execution_identity(self):
        for in_place in (False, True):
            with self.subTest(in_place=in_place), tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ):
                source = Path(temporary) / 'foundation'
                root = source / 'followup' if in_place else Path(temporary) / 'followup'
                base = base_packet()
                config = {'neuron': 'L1_N2', 'model': '/synthetic/model', 'activations': '/synthetic/archive',
                          'gpus': [], 'settings': {}, 'llm': {}, 'api_slots': str(Path(temporary) / 'slots')}
                bundle = {'neuron': {'id': 'L1_N2'}, 'criticism': [{'attack': 'original'}],
                          'anatomy': {'saved': [1, 2]}, 'ablation': {'saved': [3, 4]}}
                values = {'config.json': config, 'interpretation.json': base['initial'], 'predictions.json': {'records': []},
                          'evidence.json': {'documents': [{'id': '0'}]},
                          'artifacts.json': {'initial': base['initial'], 'predictions': {'records': []},
                                             'revision': {'criticism': [{'attack': 'original'}]},
                                             'evidence': base['catalog']}, 'status.json': {'phase': 'complete'}}
                for name, value in values.items():
                    write(source / name, value)
                write(source / 'report/paper-data.json', {'bundle': bundle})
                (source / 'report/index.html').write_text(
                    '<script id="bundle">' + json.dumps(bundle) + '</script><script>\n'
                    '// 7. Tests.\n// 8. Criticism.</script>')
                before = {name: (source / name).read_bytes() for name in (*SOURCE_FILES, 'status.json', 'report/paper-data.json')}
                # A previous publication may hardlink both files. It must remain a baseline snapshot.
                for name in ('index.html', 'paper-data.json'):
                    os.link(source / 'report' / name, Path(temporary) / name)
                client, executor = FakeClient([plan([question('first')]), plan([])]), FakeExecutor()
                kwargs = {'in_place': in_place, 'client_factory': lambda *args: client,
                          'executor_factory': lambda *args: executor}
                with patch('language.workflow.followups.input_identity', return_value={'synthetic': 1}), \
                     patch('language.workflow.followups.render', side_effect=RuntimeError('synthetic render interruption')):
                    with self.assertRaises(RuntimeError):
                        run(source, root, **kwargs)
                self.assertEqual(read(root / 'status.json')['phase'], 'failed')
                self.assertEqual(before, {name: (source / name).read_bytes() for name in before})
                legacy = read(root / 'manifest.json')
                legacy.update(implementation={'old_module.py': 'old_hash'}, config=config,
                              rounds=2, max_investigations=5)
                write(root / 'manifest.json', legacy)
                (root / 'frozen.json').unlink()
                calls_before = len(executor.calls)
                with patch('language.workflow.followups.input_identity', return_value={'synthetic': 1}):
                    if in_place:
                        for filename in ('paper-data.json', 'index.html'):
                            replace = Path.replace

                            def interrupt_promotion(path, target):
                                result = replace(path, target)
                                if Path(target) == source / 'report' / filename:
                                    raise OSError('synthetic promotion interruption')
                                return result

                            with patch.object(Path, 'replace', interrupt_promotion), \
                                 self.assertRaisesRegex(OSError, 'synthetic promotion interruption'):
                                run(source, root, **kwargs)
                            self.assertEqual(read(root / 'status.json')['phase'], 'failed')
                            expected = (before['report/index.html'] if filename == 'paper-data.json'
                                        else (root / 'report/index.html').read_bytes())
                            self.assertEqual((source / 'report/index.html').read_bytes(), expected)
                    output = run(source, root, **kwargs)
                    self.assertEqual(output, (source if in_place else root) / 'report')
                    self.assertTrue((output / 'index.html').is_file())
                    self.assertEqual(read(root / 'status.json')['phase'], 'complete')
                    self.assertEqual(read(root / 'frozen.json')['initial'], base['initial'])
                    self.assertEqual(len(executor.calls), calls_before)
                    with self.assertRaises(ValueError):
                        run(source, root, {'max_tokens': 512}, **kwargs)
                    calls_after = len(client.calls)
                    first_page = (output / 'index.html').read_bytes()
                    run(source, root, **kwargs)
                    self.assertEqual((output / 'index.html').read_bytes(), first_page)
                    self.assertEqual(len(client.calls), calls_after)
                rendered = json.loads(BUNDLE.search(first_page.decode('utf-8'))[2])
                self.assertEqual(len(rendered['criticism']), 2)
                self.assertEqual({k: v for k, v in rendered.items() if k != 'criticism'},
                                 {k: v for k, v in bundle.items() if k != 'criticism'})
                self.assertEqual(read(output / 'paper-data.json')['bundle'], rendered)
                for name in ('index.html', 'paper-data.json'):
                    self.assertEqual((Path(temporary) / name).read_bytes(), before['report/' + name])
                unchanged = [name for name in before if not in_place or not name.startswith('report/')]
                self.assertEqual({name: before[name] for name in unchanged},
                                 {name: (source / name).read_bytes() for name in unchanged})
                if in_place:
                    self.assertEqual((root / 'foundation.html').read_bytes(), before['report/index.html'])
                    # Standalone mode must not append a second investigation onto an already augmented report.
                    with self.assertRaisesRegex(ValueError, 'already contains follow-ups'):
                        run(source, Path(temporary) / 'another')
                    (root / 'foundation.html').write_bytes(before['report/index.html'] + b'changed')
                    with patch('language.workflow.followups.input_identity', return_value={'synthetic': 1}), \
                         self.assertRaisesRegex(ValueError, 'Source or settings changed'):
                        run(source, root, **kwargs)
                    (root / 'foundation.html').write_bytes(before['report/index.html'])
                with patch('language.workflow.followups.input_identity', return_value={'synthetic': 2}):
                    with self.assertRaises(ValueError):
                        run(source, root, **kwargs)
                if in_place:
                    write(source / 'route-prediction/result.json', {'rows': []})
                    html = output / 'index.html'
                    rendered = root / 'report/index.html'
                    old_page = rendered.read_text(encoding='utf-8')
                    html.write_text(route_overlay(old_page, source, costs={'total_execution_seconds': 1}),
                                    encoding='utf-8')
                    next_costs = {'total_execution_seconds': 2}
                    write(output / 'interpretation-costs.json', next_costs)
                    with patch('language.workflow.followups.input_identity', return_value={'synthetic': 1}):
                        run(source, root, **kwargs)
                        self.assertEqual(json.loads(BUNDLE.search(html.read_text(encoding='utf-8'))[2])
                                         ['interpretation_costs'], next_costs)
                        saved_bundle = json.loads(BUNDLE.search(old_page)[2])
                        saved_bundle['criticism'][-1]['evidence'] = [{'text': 'a', 'activation': 2.0}]
                        legacy_page = BUNDLE.sub(
                            lambda match: match[1] + encode_bundle(saved_bundle) + match[3], old_page, count=1)
                        html.write_text(route_overlay(legacy_page, source, costs=next_costs), encoding='utf-8')
                        rendered.unlink()
                        (root / 'report/paper-data.json').unlink()
                        run(source, root, **kwargs)
                    self.assertEqual(rendered.read_text(encoding='utf-8'), old_page)
                    self.assertEqual(len(client.calls), calls_after)
                    self.assertEqual(len(executor.calls), calls_before)
                changed = read(source / 'evidence.json')
                changed['documents'].append({'id': '1'})
                write(source / 'evidence.json', changed)
                with patch('language.workflow.followups.input_identity', return_value={'synthetic': 1}):
                    with self.assertRaises(ValueError):
                        run(source, root, **kwargs)
