"""Count the tokens at the saved corpus-wide native product extremes."""

from collections import Counter
import json
from pathlib import Path
from unicodedata import category

if __package__:
    from .build_report import byte_level_bytes
else:
    from build_report import byte_level_bytes


CENSUS_FILES = {
    'positive': ('peak-census.json', 'peak_census'),
    'negative': ('peak-census-negative.json', 'peak_census_negative'),
    'absolute_appendix': ('peak-census-absolute.json', 'peak_census_absolute'),
}
CLASSIFICATION = json.loads(
    (Path(__file__).resolve().parents[1] / 'tools/peak-token-classes.json').read_text(encoding='utf-8')
)
FUNCTION_WORDS = frozenset(CLASSIFICATION['function_words'])


def token_kind(token):
    if not token:
        return 'incomplete UTF-8 fragment'
    if token.isspace():
        return 'space / newline'
    stripped = token.strip()
    if stripped.isdecimal():
        return 'digit'
    if all(category(char)[0] in 'PS' for char in stripped):
        return 'punctuation'
    if stripped.casefold() in FUNCTION_WORDS:
        return 'function word'
    if stripped[0].isupper():
        return 'capitalised word'
    return 'other word'


def peak_token(document, pos):
    # A context decoder can attach a whole character to the final byte fragment.
    if raw_tokens := document.get('raw_tokens'):
        raw = raw_tokens[pos]
        data = byte_level_bytes(raw)
        if data is None:
            return raw
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            return ''
    token = document['tokens'][pos]
    return '' if '\ufffd' in token else token


def census_for_tail(data, pole):
    group, sign, direction = ('top_tokens', 1, 'desc') if pole == 'positive' else ('bottom_tokens', -1, 'asc')
    peaks, documents = [], set()
    # The capture builders already rank these exact extrema by product, breaking ties by token position.
    for case in data.get('examples', {}).get(group, [])[:30]:
        case_id = str(case)
        point = data['points'][case_id]
        product = point['values'][2]
        if sign * product <= 0:
            continue
        document_id, pos = str(point['document']), point['position']
        document = data['documents'][document_id]
        token = peak_token(document, pos)
        token_id = document['token_ids'][pos]
        peaks.append({'case_id': case_id, 'token_id': token_id, 'token': token or f'⟦token {token_id}⟧',
                      'kind': token_kind(token), 'product': product})
        documents.add(document_id)
    if not peaks:
        return None
    return {
        'n': len(peaks), 'pole': pole, 'population_tokens': data['n_tokens'],
        'distinct_documents': len(documents), 'selection_unit': 'token', 'peaks': peaks,
        'ranking': f'native_product_{direction}_then_global_token_asc', 'classification': CLASSIFICATION,
        'kinds': [{'kind': kind, 'n': n} for kind, n in Counter(row['kind'] for row in peaks).most_common()],
        'tokens': [{'token': token, 'n': n} for token, n in Counter(row['token'] for row in peaks).most_common()],
    }


def load_peak_censuses(report, data):
    """Keep saved censuses and recover missing signed tails from a saved report's exact extrema."""
    censuses = {}
    for pole, (filename, _) in CENSUS_FILES.items():
        path = report / filename
        census = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
        if census:
            for field, label in (('kinds', 'kind'), ('tokens', 'token')):
                if isinstance(census.get(field), dict):
                    census[field] = [{label: key, 'n': n} for key, n in census[field].items()]
        elif pole != 'absolute_appendix':
            census = census_for_tail(data, pole)
        if census:
            censuses[pole] = census
    return censuses
