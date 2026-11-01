"""Deterministic replacement sweeps and source-to-root attribution."""
import math
import time
from pathlib import Path
from common import Runtime, score_positions, select_peak, tokenize, write
from settings import request_settings

METHOD = ('Native gradient x activation: total first-order source-to-root sensitivity across downstream routes; '
          'not isolated direct neuron-to-neuron edges. Source scores are not additive causal shares.')

def record_trace(runtime, ids, offsets, state):
    return [{'position': i, 'token_id': token, 'token': runtime.tokenizer.decode([token]),
             'offset': offsets[i], **{c: float(state[c][i]) for c in ['gate', 'value', 'product']}}
            for i, token in enumerate(ids)]

def sweep(request, output, runtime=None):
    runtime = runtime or Runtime.load(request)
    output = Path(output)
    torch, rows, failures = runtime.torch, [], []
    limit = request.get('limits', {}).get('max_input_tokens', request_settings(request)['inputs']['max_written_tokens'])
    for i, replacement in enumerate(request['replacements']):
        value = replacement if isinstance(replacement, str) else replacement['text']
        label = value if isinstance(replacement, str) else replacement.get('label', value)
        prefix, suffix = request['template'].split(request.get('slot', '{slot}'))
        text = prefix + value + suffix
        try:
            ids, offsets = tokenize(runtime.tokenizer, text, limit)
            selected = score_positions(text, offsets, (len(prefix), len(prefix)+len(value)), request['score'])
            with torch.no_grad(), runtime.capture() as observed:
                runtime.forward(ids)
            trace = record_trace(runtime, ids, offsets, observed)
            winner = select_peak([trace[j] for j in selected], request['score']['reduction'])
            rows.append({'condition': i, 'status': 'complete', 'replacement': value, 'label': label,
                         'text': text, 'token_ids': ids, 'offsets': offsets, 'scored_positions': selected,
                         'selected': winner, 'trace': trace, 'native_calls': observed['calls'], 'score':request['score']})
        except Exception as exc:
            failure = {'condition': i, 'replacement': value, 'text': text,
                       'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)}
            rows.append(failure); failures.append(failure)
        write(output/'conditions.json', rows)
        if failures:
            # Model/capture failures can leave a CUDA context unusable. Preserve every unattempted condition.
            for j in range(i+1, len(request['replacements'])):
                rows.append({'condition': j, 'status': 'not_run', 'reason': 'prior condition failed'})
            break
    completed = [x for x in rows if x['status'] == 'complete']
    result = {'tool': 'replacement_sweep', 'status': 'complete' if len(completed)==len(request['replacements']) else 'incomplete',
              'neuron': request['neuron'], 'expected_conditions': len(request['replacements']),
              'completed_conditions': len(completed), 'score': request['score'], 'conditions': rows,
              'telemetry': runtime.telemetry(), 'model': getattr(runtime,'identity',None)}
    fig = {'id': request['id'], 'kind': 'sweep', 'title': request.get('title', request['id']),
           'rationale': request.get('prediction', ''), 'pole': request.get('pole'), 'template': request['template'],
           'caption': request.get('caption', f"Native product; {request['score']['location']}, {request['score']['reduction']} scoring."),
           'results': [{'tag': x['label'], 'filler': x['replacement'], 'text': x['text'],
                        'activation': x['selected']['product'], 'peak_token': x['selected']['token'],
                        'peak_index': x['selected']['position'], 'scored_tokens': x['scored_positions'],
                        'max': max(x['trace'][j]['product'] for j in x['scored_positions']),
                        'min': min(x['trace'][j]['product'] for j in x['scored_positions'])}
                       for x in completed]}
    write(output/'conditions.json', rows)
    write(output/'result.json', result)
    if result['status'] == 'complete':
        write(output/'figure.json', fig)
    return result

def native_ablation(runtime, ids, pos, target_id, gradient_logits, output=None):
    torch = runtime.torch
    with torch.no_grad(), runtime.capture() as baseline:
        a = runtime.forward(ids, pos).detach().cpu()
    with torch.no_grad(), runtime.capture() as repeat:
        repeated_logits=runtime.forward(ids,pos).detach().cpu()
    repeated=torch.equal(a,repeated_logits) and all(torch.equal(baseline[c],repeat[c]) for c in ['gate','value','product'])
    grad_difference=float((a-gradient_logits.detach().cpu()).abs().max())
    with torch.no_grad(), runtime.capture(zero_position=pos) as edited:
        b = runtime.forward(ids, pos).detach().cpu()
    unchanged_upstream = all(torch.equal(baseline[c], edited[c]) for c in ['gate','value','product'])
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite ablation logits')
    lp, lq = torch.log_softmax(a.double(), -1), torch.log_softmax(b.double(), -1)
    p, q = lp.exp(), lq.exp()
    kl = float((p*(lp-lq)).sum())
    if kl < -1e-9 or not math.isfinite(kl):
        raise ValueError('Invalid KL')
    reference = int(a.argmax()) if target_id is None else target_id
    if output is not None:
        import numpy as np
        np.savez_compressed(Path(output)/'ablation-logits.npz', baseline=a.numpy(), zeroed=b.numpy())
    def top(probs):
        return [{'token_id': int(i), 'token': runtime.tokenizer.decode([int(i)]), 'p': float(probs[i])}
                for i in probs.topk(min(8, probs.numel())).indices]
    return {'status': 'complete', 'position': pos, 'baseline_repeat_equal': repeated,
            'upstream_native_trace_equal': unchanged_upstream, 'edit': edited['edit'],
            'attribution_vs_ablation_logit_max_abs':grad_difference,
            'baseline_target_activation':float(baseline['product'][pos]),
            'native_calls': {'baseline': baseline['calls'], 'repeat':repeat['calls'], 'zeroed': edited['calls']},
            'reference_token_id': reference, 'reference_token': runtime.tokenizer.decode([reference]),
            'reference_is_requested_logit': target_id is not None,
            'delta_target_logit': float(b[reference]-a[reference]),
            'delta_target_prob': float(q[reference]-p[reference]),
            'kl': max(0., kl), 'baseline_top': top(p), 'zeroed_top': top(q),
            'logit_arrays': 'ablation-logits.npz' if output is not None else None}

def attribute(request, output, runtime=None):
    from common import position
    from relp import graph as trace_graph, METHOD as relp_method
    runtime = runtime or Runtime.load(request)
    output = Path(output)
    cfg=request_settings(request)
    if 'token_ids' in request:
        ids=list(request['token_ids']);offsets=[None]*len(ids)
    else:
        ids,offsets=tokenize(runtime.tokenizer,request['text'],cfg['inputs']['max_written_tokens'])
    pos = position(request['root']['position'], len(ids))
    following_ids=ids[pos+1:]
    ids=ids[:pos+1];offsets=offsets[:pos+1]
    target = request['root'].get('token_id') if request['root']['kind'] == 'output_logit' else None
    with runtime.torch.no_grad(), runtime.capture() as native:
        native_logits = runtime.forward(ids, pos).detach().cpu()
    graph, logits, coverage = trace_graph(runtime, ids, pos,
        request['neuron']['layer'], request['neuron']['index'], target,
        cfg['graphs']['mlp_nodes'],cfg['graphs']['embedding_nodes'],cfg['graphs']['max_edges'])
    actual_text=runtime.tokenizer.decode(ids,skip_special_tokens=False)
    graph.update(id=request['id'], text=actual_text)
    graph['meta']['native_logit_max_abs'] = float((logits-native_logits).abs().max())
    graph['meta']['native_product'] = float(native['product'][pos])
    graph['meta']['following_prediction_target_token_ids']=following_ids
    ablation = native_ablation(runtime, ids, pos, target, logits, output) if request.get('zero_target_neuron', True) else None
    graph['meta']['ablation'] = ablation
    reference = target if target is not None else int(native_logits.argmax())
    with runtime.torch.no_grad(), runtime.capture(zero_position=pos, scale=cfg['graphs']['steering_scale']) as doubled:
        double_logits=runtime.forward(ids,pos).detach().cpu()
    a=native_logits.double().softmax(-1);b=double_logits.double().softmax(-1)
    import numpy as np
    np.savez_compressed(output/'steering-logits.npz',baseline=native_logits.numpy(),doubled=double_logits.numpy())
    steering={'prompt':actual_text,'strength':cfg['graphs']['steering_scale'],'baseline':float(a[reference]),'steered':float(b[reference]),
        'reference_token_id':reference,'edit':doubled['edit'],
        'note':'Next-token probability of '+repr(runtime.tokenizer.decode([reference]))+
               ' after multiplying this neuron product by '+str(cfg['graphs']['steering_scale'])+' at token '+str(pos)+'.'}
    graph['meta']['steering']=steering
    result = {'tool':'attribute', 'status':'complete', 'neuron':request['neuron'],
        'method':relp_method, 'root':request['root'], 'root_value':graph['meta']['root_value'],
        'token_ids':ids, 'offsets':offsets, 'trace':record_trace(runtime, ids, offsets, native),
        'coverage':coverage, 'ablation':ablation, 'steering':steering, 'graph':graph,
        'telemetry':runtime.telemetry(), 'model':getattr(runtime,'identity',None)}
    write(output/'result.json', result)
    write(output/'graph.json', {'id':request['id'], 'title':request.get('title',request['id']),
        'why':request.get('prediction', ''), 'look_at':[], 'graph':graph})
    return result
