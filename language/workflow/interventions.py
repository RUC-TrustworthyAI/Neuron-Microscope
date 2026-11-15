"""Measure native point ablations against fixed same-layer controls."""
import numpy as np

from .state import read, write


def random_controls(settings, neuron, width):
    count = settings['followups']['random_controls']
    if type(count) is not int or not 1 <= count < width:
        raise ValueError('random_controls must fit the number of other neurons in the layer')
    seed = np.random.SeedSequence([settings['seed'], neuron['layer'], neuron['index']])
    candidates = [index for index in range(width) if index != neuron['index']]
    return np.random.default_rng(seed).choice(candidates, count, replace=False).tolist()


def next_token_readout(tokenizer, token_id, log_base, log_zero):
    return {
        'next_token_id': token_id,
        'next_token': tokenizer.decode([token_id]) if token_id is not None else None,
        'next_logprob_base': float(log_base[token_id]) if token_id is not None else None,
        'next_logprob_ablated': float(log_zero[token_id]) if token_id is not None else None,
    }


def measure_case(runtime, case, controls, folder):
    from model_tools import point_logits

    ids, pos = case['token_ids'], case['position']
    if not ids or pos != len(ids) - 1:
        raise ValueError(f"Invalid causal prefix for {case['case_id']}")
    baseline, trace = point_logits(runtime, ids, pos)
    target = runtime.neuron['index']
    interventions = [point_logits(runtime, ids, pos, unit, zero=True)
                     for unit in [target, *controls]]
    logits = runtime.torch.stack([baseline, *(value for value, _ in interventions)])
    if not runtime.torch.isfinite(logits).all():
        raise ValueError(f"Nonfinite logits for {case['case_id']}")
    edits = [capture['edit'] for _, capture in interventions]
    if edits[0]['before'] != float(trace['product'][pos]):
        raise RuntimeError(f"Target activation changed before intervention for {case['case_id']}")

    log_probs = logits.double().log_softmax(dim=-1)
    log_base, log_zero = log_probs[:2]
    base, zero = log_base.exp(), log_zero.exp()
    kl = (base * (log_base - log_probs[1:])).sum(dim=-1).tolist()
    kl = [max(0., value) for value in kl]
    delta = zero - base

    def top(values, key, count, largest=True):
        indices = values.topk(min(count, values.numel()), largest=largest).indices.tolist()
        return [{'token_id': index, 'token': runtime.tokenizer.decode([index]), key: float(values[index])}
                for index in indices]

    top_base = top(base, 'p', 6)
    top_id = top_base[0]['token_id']
    row = {key: value for key, value in case.items() if key != 'token_ids'}
    row.update(text=runtime.tokenizer.decode(ids, skip_special_tokens=False),
               peak_index=pos, live_product=float(trace['product'][pos]), control_neurons=controls,
               kl=kl[0], kl_controls=kl[1:], edit=edits[0],
               baseline_top_logprob_ablated=float(log_zero[top_id]),
               top_base=top_base, top_ablated=top(zero, 'p', 6),
               gains=[item for item in top(delta, 'dp', 5) if item['dp'] > 0],
               losses=[item for item in top(delta, 'dp', 5, largest=False) if item['dp'] < 0])
    row.update(next_token_readout(runtime.tokenizer, case['next_token_id'], log_base, log_zero))
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / 'logits.npz'
    temporary = raw.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, baseline=logits[0].numpy(), target=logits[1].numpy(),
                            controls=logits[2:].numpy())
    temporary.replace(raw)
    return {'schema_version': 1, 'status': 'complete', 'ablation': row,
            'edits': edits, 'logit_arrays': raw.name}


def measure_ablations(runtime, plan, folder):
    from ablation import ablation_section

    controls = plan['control_neurons']
    records = []
    start_forwards = runtime.forwards
    for ordinal, case in enumerate(plan['cases'], 1):
        case_folder = folder / 'cases' / f'{ordinal:03d}'
        path = case_folder / 'result.json'
        record = read(path) if path.is_file() and (case_folder / 'logits.npz').is_file() else None
        if record is not None:
            if (record.get('status') != 'complete' or record['ablation']['case_id'] != case['case_id']
                    or record['ablation']['control_neurons'] != controls):
                raise ValueError(f'Incompatible ablation checkpoint: {path}')
        if record is None or record.get('schema_version') != 1:
            record = measure_case(runtime, case, controls, case_folder)
            write(path, record)
        elif record['ablation']['next_token_id'] != case['next_token_id']:
            row = record['ablation']
            if (case['kind'] != 'controlled' or row['kind'] != 'controlled'
                    or row['next_token_id'] is not None or row['peak_index'] != case['position']):
                raise ValueError(f'Incompatible next-token checkpoint: {path}')
            with np.load(case_folder / 'logits.npz') as saved:
                log_base, log_zero = runtime.torch.from_numpy(
                    np.stack([saved['baseline'], saved['target']])).double().log_softmax(dim=-1)
            row.update(next_token_readout(runtime.tokenizer, case['next_token_id'], log_base, log_zero))
            write(path, record)
        records.append(record)
    section = ablation_section(records, plan)
    summary_path = folder / 'summary.json'
    if not summary_path.is_file() or runtime.forwards > start_forwards:
        write(summary_path, {'summary': section['summary'],
              'completed_case_forwards': len(records) * (2 + len(controls)),
              'forwards_this_session': runtime.forwards - start_forwards,
              'telemetry': runtime.telemetry()})
    return section
