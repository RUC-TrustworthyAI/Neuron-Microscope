"""Build a paired three-channel comparison from one retained measured sweep."""
import math
from itertools import combinations

from ..tools.common import score_positions, select_peak


CHANNELS = {'gate': 'gate', 'up': 'value', 'act': 'product'}
COMPARISONS = {
    'shared-suffix': 'Same scored token in the shared suffix',
    'matching-slot': 'Corresponding scored token within the replacement',
    'separate-peaks': 'Separate scored peaks from the same sweep',
}


def _condition(result, measured, prefix, suffix):
    row = measured.get(result.get('measurement_id'))
    if not row or row.get('status') != 'complete':
        return None
    replacement, text = row.get('replacement'), row.get('text')
    if (not isinstance(replacement, str) or replacement != result.get('filler')
            or text != result.get('text') or text != prefix + replacement + suffix):
        return None
    trace, ids = row['trace'], row['token_ids']
    offsets = [token.get('offset') for token in trace]
    if not offsets or any(offset is None for offset in offsets):
        return None

    span = [len(prefix), len(prefix) + len(replacement)]
    score, scored = row['score'], row['scored_positions']
    if score['location'] not in ('replacement_span', 'after_replacement') or score['reduction'] not in ('absmax', 'single'):
        return None
    try:
        allowed = score_positions(text, offsets, span, score)
        if not set(scored).issubset(allowed) or scored != result.get('scored_tokens'):
            return None
        selected = select_peak([trace[position] for position in scored], score['reduction'])
    except ValueError:
        return None
    anchor = selected['position']
    if (selected != row['selected'] or anchor != result.get('peak_index')
            or selected['token'] != result.get('peak_token') or selected['product'] != result.get('activation')):
        return None
    start, end = span
    changed = [index for index, (left, right) in enumerate(offsets)
               if start < end and left < right and left < end and right > start]
    return {'case_id': row['id'], 'label': row['label'], 'text': text, 'replacement_span': span,
            'replacement_positions': changed, 'score': score, 'scored_positions': scored,
            'anchor_index': anchor, 'tokens': [token['token'] for token in trace],
            'token_ids': ids, 'offsets': offsets,
            **{key: [token[channel] for token in trace] for key, channel in CHANNELS.items()}}


def _first_divergent_token(first, second):
    # Require a changed token within both causal prefixes. A strict-prefix extension
    # only adds tokens after one of the anchors and cannot establish a paired change.
    i, j = first['anchor_index'], second['anchor_index']
    for position, (a, b) in enumerate(zip(first['token_ids'][:i + 1], second['token_ids'][:j + 1])):
        if a != b:
            return position
    return None


def _anchor_kind(first, second):
    i, j = first['anchor_index'], second['anchor_index']
    ids_a, ids_b = first['token_ids'], second['token_ids']
    if ids_a[i] != ids_b[j]:
        return 'separate-peaks'

    a_start, a_end = first['replacement_span']
    b_start, b_end = second['replacement_span']
    a_left, a_right = first['offsets'][i]
    b_left, b_right = second['offsets'][j]
    if (0 <= a_left - a_end < a_right - a_end and
            (a_left - a_end, a_right - a_end) == (b_left - b_end, b_right - b_end)):
        return 'shared-suffix'

    if a_start <= a_left < a_right <= a_end and b_start <= b_left < b_right <= b_end:
        same_slot_offset = (a_left - a_start, a_right - a_start) == (b_left - b_start, b_right - b_start)
        # Different character offsets can still represent the same local token context.
        same_local_frame = i == j and i >= 2 and all(
            ids_a[k] == ids_b[k] and
            a_start <= first['offsets'][k][0] < first['offsets'][k][1] <= a_end and
            b_start <= second['offsets'][k][0] < second['offsets'][k][1] <= b_end
            for k in (i - 2, i - 1))
        if same_slot_offset or same_local_frame:
            return 'matching-slot'
    return 'separate-peaks'


def _silu(value):
    if value >= 0:
        return value / (1 + math.exp(-value))
    exponential = math.exp(value)
    return value * exponential / (1 + exponential)


def _explain(first, second, kind):
    i, j = first['anchor_index'], second['anchor_index']
    a = {channel: first[channel][i] for channel in CHANNELS}
    b = {channel: second[channel][j] for channel in CHANNELS}
    anchor = {'kind': kind, 'token_id': first['token_ids'][i] if kind != 'separate-peaks' else None,
              'token': first['tokens'][i] if kind != 'separate-peaks' else None,
              'delta_gate': b['gate'] - a['gate'], 'delta_up': b['up'] - a['up'],
              'delta_product': b['act'] - a['act']}
    values = (f"Raw gate {a['gate']:.4g} → {b['gate']:.4g}, up {a['up']:.4g} → {b['up']:.4g}, "
              f"and native product {a['act']:.4g} → {b['act']:.4g} (first → second).")
    if kind == 'separate-peaks':
        return anchor, f"{values} These values describe each condition’s own peak; the positions are compared as separate readouts."

    gate_a, gate_b = _silu(a['gate']), _silu(b['gate'])
    gate_term = (gate_b - gate_a) * (a['up'] + b['up']) / 2
    up_term = (b['up'] - a['up']) * (gate_a + gate_b) / 2
    anchor.update(gate_term=gate_term, up_term=up_term,
                  reconstruction_residual=anchor['delta_product'] - gate_term - up_term)
    return anchor, (f"{values} For SiLU(gate) × up, the symmetric change terms are "
                    f"{gate_term:+.4g} from gate and {up_term:+.4g} from up. "
                    "Rounding in the recorded precision can shift their sum from the native product change.")


def select_anatomy_pair(figures, measured, *, preferred_figure_id=None):
    """Prefer shared suffix, then matching slot, then separate peaks; break ties by sweep and contrast."""
    measured = {row['id']: row for row in measured}
    quality = {kind: rank for rank, kind in enumerate(reversed(COMPARISONS))}
    best = None
    for figure in figures:
        template = figure.get('template') or ''
        if figure.get('kind') != 'sweep' or template.count('{slot}') != 1:
            continue
        prefix, suffix = template.split('{slot}')
        rows = [_condition(result, measured, prefix, suffix) for result in figure['results']]
        for first, second in combinations((row for row in rows if row is not None), 2):
            if first['score'] != second['score']:
                continue
            divergence = _first_divergent_token(first, second)
            if divergence is None:
                continue
            kind = _anchor_kind(first, second)
            gap = abs(second['act'][second['anchor_index']] - first['act'][first['anchor_index']])
            rank = quality[kind], figure['id'] == preferred_figure_id, gap
            if best is None or rank > best[0]:
                best = rank, figure, kind, first, second, divergence
    if best is None:
        return None
    _, figure, kind, first, second, divergence = best
    anchor, explanation = _explain(first, second, kind)
    anchor['first_divergent_token_position'] = divergence
    return {'figure_id': figure['id'], 'title': figure['title'], 'template': figure['template'],
            'comparison': COMPARISONS[kind], 'rows': [first, second], 'anchor': anchor,
            'explanation': explanation,
            'selection': 'Shared suffix token first, then corresponding replacement token, then separate peaks. '
                         'Within each class, prefer the headline sweep and then the largest absolute product difference.'}
