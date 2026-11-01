"""Summarize matched neuron ablations and their random controls."""
import numpy as np


def ablation_section(records):
    rows=[r['ablation'] for r in records if r.get('status')=='complete']
    controls=sorted({n for r in rows for n in r['control_neurons']})
    return {'rows':rows,'replaces_case_ids':[str(r.get('case_id') or r.get('id') or r.get('ablation',{}).get('case_id') or r.get('ablation',{}).get('condition_id')) for r in records],
            'summary':{'n':len(rows),'expected':len(records),'controls':controls,
        'median_kl':float(np.median([r['kl'] for r in rows])) if rows else None,
        'median_kl_controls':float(np.median([k for r in rows for k in r['kl_controls']])) if rows else None}}
