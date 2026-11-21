"""Summarize matched point ablations and their random controls."""
from statistics import median


def ablation_section(records, plan):
    rows = [record['ablation'] for record in records if record['status'] == 'complete']

    def summarize(group):
        controls = [kl for row in group for kl in row['kl_controls']]
        return {'median_kl': median(row['kl'] for row in group) if group else None,
                'median_kl_controls': median(controls) if controls else None}

    coverage = {}
    for kind, requested in plan['requested'].items():
        group = [row for row in rows if row['kind'] == kind]
        coverage[kind] = {'requested': requested, 'selected': plan['realized'][kind],
                          'completed': len(group), **summarize(group)}
    return {'rows': rows, 'summary': {'n': len(rows), 'expected': sum(plan['requested'].values()),
            'controls': plan['control_neurons'], 'coverage': coverage, **summarize(rows)}}
