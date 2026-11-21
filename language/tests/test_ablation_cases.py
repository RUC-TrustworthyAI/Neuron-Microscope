import unittest

from language.workflow.ablation_cases import select_ablation_cases


def corpus_entry(document, position, value):
    return {'document': document, 'position': position, 'values': [1., value, value]}


def document(*ids):
    return {'token_ids': list(ids), 'tokens': [f' {token}' for token in ids]}


def condition(identity, ids, position, value):
    trace = [{'token': f' {token}'} for token in ids]
    return {'id': identity, 'token_ids': ids, 'trace': trace, 'status': 'complete',
            'selected': {'position': position, 'product': value, 'token': trace[position]['token']}}


def figure(identity, *condition_ids):
    return {'id': identity, 'kind': 'sweep', 'results': [{'measurement_id': row} for row in condition_ids]}


class AblationSelectionTests(unittest.TestCase):
    def test_uses_signed_rankings_and_band_boundaries_with_exact_saved_prefixes(self):
        report = {'statistics': {'product': {'min': -20., 'max': 10.}},
                  'examples': {'top_tokens': ['1', '2', '3'], 'bottom_tokens': ['4', '5']},
                  'points': {'1': corpus_entry('a', 1, 10.), '2': corpus_entry('b', 1, 9.),
                             '3': corpus_entry('c', 0, 8.), '4': corpus_entry('d', 0, -20.),
                             '5': corpus_entry('e', 0, -18.)},
                  'documents': {'a': document(1, 2, 99), 'b': document(1, 2, 100),
                                'c': document(3), 'd': document(4, 98), 'e': document(5)}}
        values = [1., -2., 0., 5., -10., 2., 5.1, -10.1, -1.5]
        ids = [10 + i for i in range(len(values))]
        contexts = {'documents': {str(i): document(i, 1000 + i) for i in ids},
                    'points': {f'spectrum-{i}': corpus_entry(str(i), 0, value)
                               for i, value in zip(ids, values)}}
        sample = {'points': [{'id': f'spectrum-{i}', 'dataset_token_index': i} for i in ids]}
        predictions = {'records': [{'id': 'spectrum-13', 'group': 'label-x'}]}
        rows = [condition('duplicate-natural', [1, 2, 111], 1, 9.),
                condition('chosen-a', [60, 61, 62], 1, .1), condition('chosen-b', [70, 71], 0, -.2),
                condition('duplicate-control', [60, 61, 63], 1, .1),
                condition('other-a', [80, 81], 0, .5), condition('other-b', [90, 91], 0, -.5)]
        figures = [figure('other', 'other-a', 'other-b'),
                   figure('headline', 'duplicate-natural', 'chosen-a', 'chosen-b', 'duplicate-control')]
        plan = select_ablation_cases(sample, contexts, rows, figures, data=report, predictions=predictions,
                                     preferred_figure_id='headline', natural_top=2, natural_bottom=2,
                                     medium=10, weak=10, controlled=2)
        cases = plan['cases']
        self.assertEqual(plan['realized'], {'strong-positive': 2, 'strong-negative': 2, 'medium': 3,
                                            'weak': 4, 'controlled': 2})
        self.assertEqual([row['source_id'] for row in cases[:4]], ['1', '3', '4', '5'])
        self.assertEqual(cases[0]['token_ids'], [1, 2])
        self.assertEqual(cases[0]['position'], 1)
        self.assertEqual(cases[0]['next_token_id'], 99)
        self.assertIsNone(cases[1]['next_token_id'])
        self.assertEqual({row['saved_product'] for row in cases if row['kind'] == 'medium'}, {5., -10., 2.})
        self.assertEqual({row['saved_product'] for row in cases if row['kind'] == 'weak'}, {1., -2., 0., -1.5})
        self.assertEqual(next(row for row in cases if row['source_id'] == 'spectrum-13')['group'], 'label-x')
        self.assertEqual(plan['controlled_sweep'], 'headline')
        self.assertEqual([row['condition_id'] for row in cases[-2:]], ['chosen-a', 'chosen-b'])
        self.assertEqual([row['next_token_id'] for row in cases[-2:]], [62, 71])
        self.assertEqual(cases[-2]['token_ids'], [60, 61])

    def test_fills_one_sweep_and_prefers_natural_document_and_label_diversity(self):
        points = {'a': corpus_entry('shared', 0, 2.), 'b': corpus_entry('shared', 1, 3.),
                  'c': corpus_entry('other', 0, 3.), 'd': corpus_entry('negative', 0, -2.),
                  'e': corpus_entry('other-negative', 0, -3.)}
        documents = {'shared': document(1, 2), 'other': document(3),
                     'negative': document(4), 'other-negative': document(5)}
        contexts = {'points': points, 'documents': documents}
        sample = {'points': [{'id': identity} for identity in points]}
        predictions = {'records': [{'id': identity, 'group': 'group-x' if identity in ('a', 'b', 'c') else 'group-y'}
                                    for identity in points]}
        data = {'statistics': {'product': {'min': -10., 'max': 10.}}}
        rows = [condition('preferred-1', [30], 0, 1.), condition('full-1', [31], 0, 2.),
                condition('full-2', [32], 0, -1.)]
        figures = [figure('preferred', 'preferred-1'), figure('full', 'full-1', 'full-2')]
        arguments = {'data': data, 'predictions': predictions, 'preferred_figure_id': 'preferred',
                     'natural_top': 0, 'natural_bottom': 0, 'medium': 3, 'weak': 1, 'controlled': 2, 'seed': 7}
        plan = select_ablation_cases(sample, contexts, rows, figures, **arguments)
        reordered = {'points': dict(reversed(list(points.items()))), 'documents': documents}
        self.assertEqual(plan, select_ablation_cases(sample, reordered, rows, figures, **arguments))
        medium = [row for row in plan['cases'] if row['kind'] == 'medium']
        self.assertEqual(len({row['document_id'] for row in medium}), 3)
        self.assertEqual({row['group'] for row in medium}, {'group-x', 'group-y'})
        self.assertEqual(plan['realized']['weak'], 0)
        self.assertEqual(plan['controlled_sweep'], 'full')
        self.assertEqual([row['condition_id'] for row in plan['cases'][-2:]], ['full-1', 'full-2'])
        self.assertTrue(all(row['next_token_id'] is None for row in plan['cases'][-2:]))


if __name__ == '__main__':
    unittest.main()
