"""Offline accounting for integrated and older follow-up checkpoints."""
import tempfile
import unittest
from pathlib import Path

from language.workflow.render import costs
from language.workflow.state import read, write


class WorkflowCostsTests(unittest.TestCase):
    def test_followup_costs_use_saved_stages_and_union_recorded_intervals(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            foundation_timeline = [
                {'resource': 'execution', 'stage': 0, 'start': 0, 'end': 6},
                {'resource': 'gpu', 'stage': 3, 'start': 2, 'end': 5},
            ]
            write(root / 'timeline.json', foundation_timeline)
            write(root / 'api/initial/timing.json',
                  {'stage': 2, 'start': 0, 'end': 6, 'first_chunk': 1, 'last_chunk': 5})
            write(root / 'api/initial/billing.json', {'estimated_usd': .25})
            original = costs(root, 'L1_N2', 2)
            self.assertEqual(original['completed_steps'], 5)
            self.assertEqual(len(original['steps']), 5)
            self.assertEqual(original['total_execution_seconds'], 6)

            followup = root / 'followup'
            for name, stage, start, end, first, last, usd in (
                    ('plan', 1, 4, 7, 4.5, 6, .5),
                    ('assessment', 2, 5, 9, 5.5, 8, .75)):
                write(followup / 'api' / name / 'timing.json',
                      {'stage': stage, 'start': start, 'end': end,
                       'first_chunk': first, 'last_chunk': last})
                write(followup / 'api' / name / 'billing.json', {'estimated_usd': usd})
            write(followup / 'timeline.json', [
                {'resource': 'execution', 'stage': 0, 'start': 6, 'end': 10},
                {'resource': 'gpu', 'stage': 3, 'start': 4, 'end': 8},
            ])
            write(root / 'timeline.json', [*foundation_timeline,
                  {'resource': 'execution', 'stage': 6, 'start': 6, 'end': 10}])
            result = costs(root, 'L1_N2', 2)
            self.assertEqual(result['completed_steps'], 6)
            self.assertEqual(result['steps'][:5], original['steps'])
            extra = result['steps'][5]
            self.assertEqual(extra['name'], 'Follow-up investigation')
            self.assertEqual(extra['api_requests'], 2)
            self.assertEqual(extra['api_usd'], 1.25)
            self.assertEqual(extra['gpu_seconds'], 4)
            self.assertAlmostEqual(extra['gpu_card_hours'], 8 / 3600)
            self.assertEqual(extra['api_stream_seconds'], 3.5)
            self.assertEqual(result['total_api_requests'], 3)
            self.assertEqual(result['total_api_usd'], 1.5)
            self.assertEqual(result['total_gpu_seconds'], 6)
            self.assertAlmostEqual(result['total_gpu_card_hours'], 12 / 3600)
            self.assertEqual(result['total_api_stream_seconds'], 7)
            self.assertEqual(result['total_api_request_seconds'], 9)
            self.assertEqual(result['total_execution_seconds'], 10)
            self.assertEqual(read(followup / 'api/plan/timing.json')['stage'], 1)
            self.assertEqual(read(followup / 'timeline.json')[1]['stage'], 3)

            # Older follow-ups lack execution spans. API timestamps must not be
            # treated as a substitute for unrecorded wall-clock execution time.
            (followup / 'timeline.json').unlink()
            write(root / 'timeline.json', foundation_timeline)
            legacy = costs(root, 'L1_N2', 2)
            self.assertEqual(legacy['total_execution_seconds'], 6)
            self.assertEqual(legacy['total_elapsed_seconds'], 6)
            self.assertEqual(legacy['steps'][5]['gpu_seconds'], 0)
            self.assertEqual(legacy['total_api_usd'], 1.5)

            # A completed round may have no API or measurement records.
            empty = root / 'empty'
            write(empty / 'timeline.json', foundation_timeline)
            write(empty / 'followup/status.json', {'phase': 'complete'})
            self.assertEqual(len(costs(empty, 'L1_N2', 2)['steps']), 6)


if __name__ == '__main__':
    unittest.main()
