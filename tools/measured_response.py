"""Separate measured response signs from semantic descriptions."""
import collections
import math


def response_pole(value):
    if value is None or not math.isfinite(float(value)):
        return 'unknown'
    return 'positive' if value > 0 else 'negative' if value < 0 else 'zero'


def summary(values):
    values = [v for v in values if response_pole(v) != 'unknown']
    counts = collections.Counter(response_pole(v) for v in values)
    return {'n': len(values), **{k: counts[k] for k in ('positive', 'negative', 'zero')},
            'min': min(values, default=None), 'max': max(values, default=None)}


def describe_group(group, values):
    result = {k: v for k, v in group.items() if k not in ('pole', 'response_summary')}
    stats = summary(values)
    # A mixed semantic group has no single measured sign.
    result['pole'] = ('positive' if stats['positive'] and not stats['negative'] else
                      'negative' if stats['negative'] and not stats['positive'] else 'either')
    result['response_summary'] = stats
    return result


def semantic_definition(group):
    return {k: v for k, v in group.items()
            if k not in ('pole', 'response_summary', 'saved_examples', 'n_observed_members')}


def annotate_spectrum(spectrum):
    values = collections.defaultdict(list)
    for point in spectrum.get('points', []):
        point['response_pole'] = response_pole(point.get('activation'))
        values[point['group']].append(point.get('activation'))
    spectrum['groups'] = [describe_group(g, values[g['id']]) for g in spectrum.get('groups', [])]
    return spectrum


def mixed_batches(points, size):
    """Distribute the observed activation range across every annotation batch."""
    ordered = sorted(points, key=lambda p: (p['activation'], str(p['id'])))
    count = max(1, math.ceil(len(ordered) / size))
    return [ordered[i::count] for i in range(count)]
