"""Reuse graph ablations and measure same-layer random-neuron controls."""
import numpy as np
from .state import read, write


def matched_ablation(runtime, results, folder, settings):
    from followups import ablation_section
    torch = runtime.torch
    width = runtime.model.config.intermediate_size
    seed = np.random.SeedSequence([settings['seed'], runtime.neuron['layer'], runtime.neuron['index']])
    controls = np.random.default_rng(seed).choice(
        [i for i in range(width) if i != runtime.neuron['index']],
        settings['followups']['random_controls'], replace=False).tolist()
    records = []
    for index, result in enumerate(results, 1):
        path = folder / str(index) / 'matched-ablation.json'
        if path.exists():
            records.append(read(path))
            continue
        ids, pos = result['token_ids'], result['graph']['position']
        arrays = np.load(path.parent / 'ablation-logits.npz')
        baseline, zeroed = (torch.from_numpy(arrays[k]) for k in ('baseline', 'zeroed'))
        arrays.close()
        reference = baseline.double().log_softmax(-1)
        p, q = reference.exp(), zeroed.double().softmax(-1)
        counterfactuals = []
        for unit in controls:
            with torch.no_grad(), runtime.capture(zero_position=pos, neuron_index=unit):
                counterfactuals.append(runtime.forward(ids, pos).detach().cpu())
        np.savez_compressed(path.parent / 'control-logits.npz',
                            controls=np.stack([x.numpy() for x in counterfactuals]))
        delta = q - p
        row = {'case_id': result['graph']['id'], 'condition_id': result['graph']['id'],
               'kind': 'graph', 'peak_index': pos, 'peak_token': result['trace'][pos]['token'],
               'text': result['graph']['text'], 'saved_product': result['trace'][pos]['product'],
               'live_product': result['ablation']['baseline_target_activation'],
               'control_neurons': controls, 'kl': result['ablation']['kl'],
               'kl_controls': [float((p * (reference - x.double().log_softmax(-1))).sum()) for x in counterfactuals],
               'top_base': result['ablation']['baseline_top'], 'top_ablated': result['ablation']['zeroed_top'],
               'gains': [{'token': runtime.tokenizer.decode([int(i)]), 'dp': float(delta[i])}
                         for i in delta.topk(5).indices],
               'losses': [{'token': runtime.tokenizer.decode([int(i)]), 'dp': float(delta[i])}
                         for i in (-delta).topk(5).indices], 'edit': result['ablation']['edit']}
        record = {'status': 'complete', 'ablation': row}
        write(path, record)
        records.append(record)
    return ablation_section(records)
