"""Execute fixed specifications using existing measurements, without text extraction."""
from contextlib import ExitStack
from pathlib import Path

from .measure import Measurements
from .state import lock


def contrast_items(spec, tokenizer, identity, max_tokens):
    from common import score_positions, tokenize

    conditions = spec['conditions']
    if len({row['label'] for row in conditions}) != len(conditions):
        raise ValueError('Condition labels must be distinct')
    if len({row['replacement'] for row in conditions}) != len(conditions):
        raise ValueError('Conditions must have distinct replacements')
    mode = spec['score_location']
    score = ({'location': 'token_position', 'position': -1, 'reduction': 'single'}
             if mode == 'last_token' else {'location': mode, 'reduction': 'absmax'})
    items = []
    for frame_index, frame in enumerate(spec['frames'], 1):
        prefix, suffix = frame['prefix'], frame['suffix']
        if mode == 'after_replacement' and not suffix.strip():
            raise ValueError('after_replacement needs a nonempty suffix')
        frame_items, token_lists = [], []
        for index, condition in enumerate(conditions, 1):
            replacement = condition['replacement']
            if mode == 'replacement_span' and not replacement:
                raise ValueError('replacement_span needs nonempty replacements')
            text = prefix + replacement + suffix
            span = [len(prefix), len(prefix) + len(replacement)]
            ids, offsets = tokenize(tokenizer, text, max_tokens)
            positions = score_positions(text, offsets, span, score)
            frame_items.append({'id': f'{identity}-f{frame_index}-c{index}',
                'label': condition['label'], 'text': text, 'span': span, 'score': score,
                'frame': frame_index, 'condition': index, 'positions': positions})
            token_lists.append(ids)
        common = 0
        for tokens in zip(*token_lists):
            if len(set(tokens)) != 1:
                break
            common += 1
        for item in frame_items:
            item['positions'] = [position for position in item['positions'] if position >= common]
            if not item['positions']:
                raise ValueError('The scored region must include differing or later tokens')
        items.extend(frame_items)
    return items


class Executor:
    """Load one GPU runtime lazily; review and corpus queries remain CPU-only."""
    def __init__(self, config, root, timeline, options, excluded_documents):
        self.config, self.root, self.timeline = config, Path(root), timeline
        self.options, self.excluded_documents = options, excluded_documents
        self.stack, self.runtime, self.tokenizer = ExitStack(), None, None

    def __enter__(self):
        return self

    def __exit__(self, *error):
        try:
            if self.runtime is not None:
                import gc
                torch = self.runtime.torch
                self.runtime = None
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            self.stack.close()

    def validate_spec(self, action, spec, identity):
        if action == 'corpus':
            from .followup_corpus import validate_spec
            validate_spec(spec)
        if action == 'contrast':
            if self.tokenizer is None:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.config['model'], local_files_only=True)
            contrast_items(spec, self.tokenizer, identity, self.options['max_tokens'])

    def execute(self, action, spec, identity):
        if action == 'corpus':
            from .followup_corpus import lookup
            return lookup(self.config, spec, seed=self.options['seed'],
                          limit=self.options['natural_per_group'],
                          exclude_document_ids=self.excluded_documents)
        from common import Runtime, select_peak
        items = contrast_items(spec, self.tokenizer, identity, self.options['max_tokens'])
        if self.runtime is None:
            layer, index = self.config['neuron'][1:].split('_N')
            with ExitStack() as loading:
                loading.enter_context(lock(self.config['gpu_lock']))
                self.runtime = Runtime.load({'model': self.config['model'],
                    'neuron': {'layer': int(layer), 'index': int(index)}, 'settings': self.config['settings']})
                self.stack.enter_context(loading.pop_all())
            from .state import write
            write(self.root / 'runtime.json', {'identity': self.runtime.identity, 'versions': self.runtime.versions})
        measure = Measurements(self.runtime, self.root / identity, self.timeline,
                               batch_size=1, batch_tokens=self.options['max_tokens'])
        rows = measure.texts(items, 'conditions')
        for item, row in zip(items, rows):
            row['scored_positions'] = item['positions']
            row['selected'] = select_peak([row['trace'][p] for p in item['positions']], item['score']['reduction'])
            row.update(frame=item['frame'], condition=item['condition'])
        references = {row['frame']: row['selected']['product'] for row in rows if row['condition'] == 1}
        for row in rows:
            row['delta_from_reference'] = row['selected']['product'] - references[row['frame']]
        return {'kind': 'contrast', 'reference_condition': 1, 'rows': rows,
                'selection': 'All declared conditions; scored positions exclude the identical token prefix.'}
