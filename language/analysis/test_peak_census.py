import json
from pathlib import Path
import tempfile
import unittest

from peak_census import census_for_tail, load_peak_censuses


class PeakCensusTests(unittest.TestCase):
    def test_saved_positive_tail_and_calculated_negative_tail(self):
        tokens = [' the', ' ', '(', '0', '', 'place']
        data = {
            'n_tokens': len(tokens),
            'examples': {'top_tokens': ['0'], 'bottom_tokens': ['1', '2', '3', '4', '5', '0']},
            'points': {str(i): {'document': 'doc', 'position': i, 'values': [0, 0, product]}
                       for i, product in enumerate([1, -4, -4, -3, -2, 0])},
            'documents': {'doc': {'tokens': tokens, 'token_ids': list(range(len(tokens)))}},
        }
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            (report / 'peak-census.json').write_text(json.dumps({'n': 1, 'tokens': {' the': 1}}))
            (report / 'peak-census-negative.json').write_text('null')
            census = load_peak_censuses(report, data)
        self.assertEqual(census['positive']['tokens'], [{'token': ' the', 'n': 1}])
        negative = census['negative']
        self.assertEqual(negative['n'], 4)
        self.assertEqual([row['case_id'] for row in negative['peaks']], ['1', '2', '3', '4'])
        self.assertEqual([row['kind'] for row in negative['peaks']],
                         ['space / newline', 'punctuation', 'digit', 'incomplete UTF-8 fragment'])
        self.assertEqual([row['token'] for row in negative['peaks']][:3], [' ', '(', '0'])

    def test_split_bytes_are_counted_by_native_token_identity(self):
        data = {
            'n_tokens': 3, 'examples': {'top_tokens': ['0', '1', '2']},
            'points': {str(i): {'document': 'doc', 'position': i, 'values': [0, 0, 3 - i]} for i in range(3)},
            'documents': {'doc': {'tokens': ['©', 'é', ' Café'], 'token_ids': [16, 16, 17],
                                  'raw_tokens': ['©', '©', 'ĠCafÃ©']}},
        }
        census = census_for_tail(data, 'positive')
        self.assertEqual(census['tokens'], [{'token': '⟦token 16⟧', 'n': 2}, {'token': ' Café', 'n': 1}])
        self.assertEqual(census['kinds'], [{'kind': 'incomplete UTF-8 fragment', 'n': 2},
                                          {'kind': 'capitalised word', 'n': 1}])
