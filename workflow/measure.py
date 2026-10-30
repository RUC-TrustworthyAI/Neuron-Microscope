"""One resident model; padded batch forwards preserve each input and native MLP product."""
import hashlib
import json
from pathlib import Path
from .state import read, write


def input_key(item):
    return hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Measurements:
    def __init__(self, runtime, root, timeline, batch_size=8, batch_tokens=8192):
        self.runtime, self.root, self.timeline = runtime, Path(root), timeline
        self.batch_size, self.batch_tokens = batch_size, batch_tokens
        self.root.mkdir(parents=True, exist_ok=True)

    def texts(self, items, name):
        from common import tokenize, score_positions, select_peak
        saved, pending = {}, []
        for item in items:
            key = input_key({'text': item['text'], 'score': item['score'], 'span': item.get('span')})
            path = self.root / 'traces' / (key + '.json')
            if path.exists():
                saved[item['id']] = dict(read(path), id=item['id'], label=item.get('label', ''))
                continue
            ids, offsets = tokenize(self.runtime.tokenizer, item['text'], None)
            selected = score_positions(item['text'], offsets, item.get('span', (0, len(item['text']))), item['score'])
            pending.append((item, key, ids, offsets, selected))
        # Length sorting limits padded work while output ordering remains unchanged.
        pending.sort(key=lambda row: len(row[2]))
        while pending:
            batch = [pending.pop(0)]
            while pending and len(batch) < self.batch_size and len(pending[0][2]) * (len(batch) + 1) <= self.batch_tokens:
                batch.append(pending.pop(0))
            with self.timeline.span(name, 'gpu', 3):
                channels = self._forward([row[2] for row in batch])
            for (item, key, ids, offsets, selected), observed in zip(batch, channels):
                trace = [{'position': i, 'token_id': token, 'token': self.runtime.tokenizer.decode([token]),
                          'offset': offsets[i], **{c: observed[c][i] for c in ('gate', 'value', 'product')}}
                         for i, token in enumerate(ids)]
                result = {'text': item['text'], 'token_ids': ids, 'trace': trace, 'scored_positions': selected,
                          'selected': select_peak([trace[p] for p in selected], item['score']['reduction']),
                          'score': item['score'], 'status': 'complete'}
                write(self.root / 'traces' / (key + '.json'), result)
                saved[item['id']] = dict(result, id=item['id'], label=item.get('label', ''))
        result = [saved[item['id']] for item in items]
        write(self.root / (name + '.json'), result)
        return result

    def _forward(self, inputs):
        runtime, torch = self.runtime, self.runtime.torch
        width = max(map(len, inputs))
        pad = runtime.tokenizer.pad_token_id
        if pad is None:
            pad = runtime.tokenizer.eos_token_id
        device = runtime.model.model.embed_tokens.weight.device
        ids = torch.tensor([x + [pad] * (width - len(x)) for x in inputs], device=device)
        mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in inputs], device=device)
        mlp = runtime.model.model.layers[runtime.neuron['layer']].mlp
        index, state, hooks = runtime.neuron['index'], {}, []
        def capture(key, pre=False):
            def hook(module, args, output=None):
                tensor = args[0] if pre else output
                state[key] = tensor[:, :, index].detach().float().cpu()
            return hook
        try:
            hooks.append(mlp.gate_proj.register_forward_hook(capture('gate')))
            hooks.append(mlp.up_proj.register_forward_hook(capture('value')))
            hooks.append(mlp.down_proj.register_forward_pre_hook(capture('product', True)))
            with torch.no_grad():
                runtime.model.model(input_ids=ids, attention_mask=mask, use_cache=False)
            runtime.forwards += 1
            for device in runtime.settings['devices']:
                torch.cuda.synchronize(device)
            return [{key: values[i, :len(tokens)].tolist() for key, values in state.items()}
                    for i, tokens in enumerate(inputs)]
        finally:
            for hook in hooks:
                hook.remove()

    def controls(self, plans):
        from .contracts import control_request
        results, figures = [], []
        for plan in plans:
            request = control_request(plan)
            prefix, suffix = request['template'].split('{slot}')
            items = [{'id': plan['id'] + ':' + str(i), 'label': replacement['label'],
                      'text': prefix + replacement['text'] + suffix,
                      'span': [len(prefix), len(prefix) + len(replacement['text'])], 'score': request['score']}
                     for i, replacement in enumerate(request['replacements'])]
            rows = self.texts(items, 'control-' + plan['id'])
            for row, replacement in zip(rows, request['replacements']):
                row['replacement'] = replacement['text']
                row['expected'] = replacement.get('expected')
            results.extend(rows)
            figures.append({'id': plan['id'], 'kind': 'sweep', 'title': plan['title'],
                'rationale': plan['rationale'], 'template': plan['template'],
                'caption': f"Signed product, {plan['score']} with {plan['reduction']} reduction.",
                'results': [{'tag': row['label'], 'filler': replacement['text'], 'text': row['text'],
                             'activation': row['selected']['product'], 'peak_token': row['selected']['token'],
                             'peak_index': row['selected']['position'], 'scored_tokens': row['scored_positions'],
                             'max': max(row['trace'][i]['product'] for i in row['scored_positions']),
                             'min': min(row['trace'][i]['product'] for i in row['scored_positions'])}
                            for row, replacement in zip(rows, request['replacements'])]})
        write(self.root / 'figures.json', figures)
        return results, figures


def summarized(row):
    return {'id': row['id'], 'label': row.get('label'), 'text': row['text'], 'score': row['score'],
            'expected': row.get('expected'),
            'selected': row['selected'], 'max': max(x['product'] for x in row['trace']),
            'min': min(x['product'] for x in row['trace']),
            'strongest_tokens': sorted(row['trace'], key=lambda x: -abs(x['product']))[:8]}


def search_regions(proposals, measurements):
    by_id = {row['id']: row for row in measurements}
    return [{**{k: region[k] for k in ('id', 'name', 'pole')},
             'description': region['rationale'],
             'candidates': [by_id[candidate['id']] for candidate in region['candidates']], 'controls': []}
            for region in proposals]
