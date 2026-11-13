import copy
import unittest

from language.workflow.anatomy_pairs import select_anatomy_pair


def condition(identity, pieces, *, first_slot_token, first_suffix_token, anchor, gate, up, product,
              location='after_replacement'):
    trace = []
    start = 0
    for index, (piece, token_id) in enumerate(pieces):
        end = start + len(piece)
        trace.append({'position': index, 'token': piece, 'token_id': token_id, 'offset': [start, end],
                      'gate': gate if index == anchor else 0., 'value': up if index == anchor else 0.,
                      'product': product if index == anchor else 0.})
        start = end
    scored = [anchor] if location == 'after_replacement' else list(range(first_slot_token, first_suffix_token))
    replacement = ''.join(piece for piece, _ in pieces[first_slot_token:first_suffix_token])
    return {'id': identity, 'status': 'complete', 'text': ''.join(piece for piece, _ in pieces),
            'token_ids': [token_id for _, token_id in pieces], 'trace': trace, 'label': identity,
            'replacement': replacement, 'selected': trace[anchor], 'scored_positions': scored,
            'score': {'location': location, 'reduction': 'single' if location == 'after_replacement' else 'absmax'}}


def sweep(identity, template, *rows):
    return {'id': identity, 'title': identity, 'kind': 'sweep', 'template': template,
            'results': [{'measurement_id': row['id'], 'text': row['text'], 'filler': row['replacement'],
                         'peak_index': row['selected']['position'], 'peak_token': row['selected']['token'],
                         'activation': row['selected']['product'], 'scored_tokens': row['scored_positions']}
                        for row in rows]}


class AnatomyPairsTests(unittest.TestCase):
    def test_aligns_the_shared_suffix_after_unequal_replacements_and_preserves_native_values(self):
        first = condition('short', [('P', 1), (' salt', 2), (',', 3)], first_slot_token=1,
                          first_suffix_token=2, anchor=2, gate=0., up=2., product=0.)
        second = condition('long', [('P', 1), (' spring', 4), (' water', 5), (',', 3)], first_slot_token=1,
                           first_suffix_token=3, anchor=3, gate=2., up=4., product=7.0625)
        figures = [sweep('suffix', 'P{slot},', first, second)]
        pair = select_anatomy_pair(figures, [first, second])
        self.assertEqual(pair['anchor']['kind'], 'shared-suffix')
        self.assertEqual(pair['anchor']['token_id'], 3)
        self.assertEqual(pair['anchor']['first_divergent_token_position'], 1)
        self.assertEqual([row['anchor_index'] for row in pair['rows']], [2, 3])
        self.assertEqual([row['replacement_positions'] for row in pair['rows']], [[1], [1, 2]])
        self.assertEqual(pair['rows'][1]['token_ids'], second['token_ids'])
        self.assertEqual(pair['rows'][1]['offsets'], [token['offset'] for token in second['trace']])
        self.assertEqual(pair['rows'][1]['act'], [0., 0., 0., 7.0625])
        self.assertEqual(pair['anchor']['delta_product'], 7.0625)
        self.assertGreater(pair['anchor']['gate_term'], pair['anchor']['up_term'])
        self.assertAlmostEqual(sum(pair['anchor'][key] for key in ('gate_term', 'up_term', 'reconstruction_residual')),
                               7.0625)
        self.assertIn('Rounding in the recorded precision', pair['explanation'])

    def test_prefers_matching_anchor_quality_before_the_headline_sweep(self):
        def material(identity, word, material_id, preposition, prep_id, value):
            return condition(identity, [('P', 1), (' with', 2), (word, material_id), (' for', 9), (' use', 10),
                                        (preposition, prep_id), ('.', 3)],
                             first_slot_token=1, first_suffix_token=6, anchor=5, gate=2., up=1.,
                             product=value, location='replacement_span')

        salt = material('salt', ' salt', 4, ' to', 6, 3.)
        water = material('water', ' water', 5, ' to', 6, 2.)
        other = material('other', ' oil', 8, ' for', 7, 20.)
        preferred = sweep('headline', 'P{slot}.', salt, other)
        matching = sweep('matching', 'P{slot}.', salt, water)
        pair = select_anatomy_pair([preferred, matching], [salt, water, other], preferred_figure_id='headline')
        self.assertEqual(pair['figure_id'], 'matching')
        self.assertEqual(pair['anchor']['kind'], 'matching-slot')
        self.assertEqual(pair['anchor']['token_id'], 6)
        fallback = select_anatomy_pair([preferred], [salt, other])
        self.assertEqual(fallback['anchor']['kind'], 'separate-peaks')
        self.assertIsNone(fallback['anchor']['token_id'])
        self.assertNotIn('gate_term', fallback['anchor'])

        one_shared = condition('one-shared', [('P', 1), (' with', 2), (' water', 5), (' into', 11), (' use', 10),
                                               (' to', 6), ('.', 3)],
                               first_slot_token=1, first_suffix_token=6, anchor=5, gate=2., up=1.,
                               product=2., location='replacement_span')
        weak_match = sweep('one-shared', 'P{slot}.', salt, one_shared)
        self.assertEqual(select_anatomy_pair([weak_match], [salt, one_shared])['anchor']['kind'], 'separate-peaks')

    def test_skips_future_only_differences_and_inconsistent_saved_anchors(self):
        first = condition('salt', [('P', 1), (' with', 2), (' salt', 3), ('.', 4)], first_slot_token=1,
                          first_suffix_token=3, anchor=1, gate=2., up=1., product=2., location='replacement_span')
        second = condition('water', [('P', 1), (' with', 2), (' water', 5), ('.', 4)], first_slot_token=1,
                           first_suffix_token=3, anchor=1, gate=2., up=1., product=2., location='replacement_span')
        figure = sweep('future-only', 'P{slot}.', first, second)
        self.assertIsNone(select_anatomy_pair([figure], [first, second]))

        later = condition('later', [('P', 1), (' with', 2), (' water', 5), (' with', 2), ('.', 4)],
                          first_slot_token=1, first_suffix_token=4, anchor=3, gate=2., up=2., product=4.,
                          location='replacement_span')
        self.assertIsNone(select_anatomy_pair([sweep('strict-prefix', 'P{slot}.', first, later)], [first, later]))

        second = condition('water', [('P', 1), (' in', 6), (' water', 5), ('.', 4)], first_slot_token=1,
                           first_suffix_token=3, anchor=1, gate=2., up=2., product=4., location='replacement_span')
        figure = sweep('inconsistent', 'P{slot}.', first, second)
        invalid = copy.deepcopy(second)
        invalid['selected']['product'] = 50.
        self.assertIsNone(select_anatomy_pair([figure], [first, invalid]))
        self.assertIsNotNone(select_anatomy_pair([figure], [first, second]))


if __name__ == '__main__':
    unittest.main()
