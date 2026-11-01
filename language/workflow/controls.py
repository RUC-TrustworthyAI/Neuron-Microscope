"""Design and measure controlled comparisons, then improve them once."""
from . import contracts, prompts
from .measure import summarized
from .state import write


def choose_versions(original, revised, corpus, explanation):
    supported = {r['predicted_pole'] for r in explanation['response_rules']}
    selected = []
    revised_by_id = {figure['id']: figure for figure in revised}
    for before in original:
        after = revised_by_id[before['id']]
        intended = {r.get('expected') for r in before['results']} & supported
        def strength(figure):
            return max((max(0., sign * row['activation']) / (abs(corpus[field]) or 1.)
                        for pole, sign, field in [('positive', 1, 'max'), ('negative', -1, 'min')]
                        if pole in (intended or supported)
                        for row in figure['results'] if row.get('expected') in (None, pole)), default=0.)
        selected.append(max((before, after), key=strength))
    return selected


def feedback(rows, corpus):
    cases = []
    for row in rows:
        value = row['selected']['product']
        extreme = corpus['max'] if value > 0 else corpus['min']
        expected = row.get('expected')
        cases.append({'id': row['id'], 'label': row['label'], 'text': row['text'],
                      'expected': expected, 'activation': value,
                      'peak': {key: row['selected'][key] for key in ('position', 'token', 'product')},
                      'relative_strength': abs(value) / abs(extreme) if extreme else None,
                      'wrong_direction': (expected == 'positive' and value <= 0) or
                                         (expected == 'negative' and value >= 0)})
    return {'corpus_extremes': {key: corpus[key] for key in ('min', 'max')}, 'conditions': cases}


def check(plan, tokenizer=None):
    for control in plan['controls']:
        request = contracts.control_request(control)
        if tokenizer is None:
            continue
        from common import tokenize, score_positions
        prefix, suffix = request['template'].split('{slot}')
        conditions = []
        for replacement in request['replacements']:
            text = prefix + replacement['text'] + suffix
            ids, offsets = tokenize(tokenizer, text, None)
            try:
                positions = score_positions(text, offsets,
                    [len(prefix), len(prefix) + len(replacement['text'])], request['score'])
            except ValueError as error:
                raise ValueError(f"Test {control['id']}, condition {replacement['label']}: {error}. "
                    'Use absmax consistently when the scored span contains multiple tokens, '
                    'or make the scored suffix exactly one token in every condition. '
                    'Preserve the comparison being tested.') from error
            conditions.append((replacement['label'], ids, positions))
        first_change = 0
        for tokens in zip(*(ids for _, ids, _ in conditions)):
            if len(set(tokens)) != 1:
                break
            first_change += 1
        for label, _, positions in conditions:
            if not any(position >= first_change for position in positions):
                raise ValueError(f'Test {control["id"]}, condition {label}: the scored span ends '
                    'before the conditions differ. Use after_replacement with a nonempty suffix '
                    'or include a changed token in every scored span. Preserve the comparison being tested.')


def design(client, explanation, packet, search_rows, tokenizer=None):
    natural, documents = [], set()
    for sign in (1, -1):
        count = 0
        for row in sorted(packet['samples'], key=lambda row: -sign * row['anchor']['product']):
            if sign * row['anchor']['product'] <= 0 or row['document'] in documents:
                continue
            natural.append(row)
            documents.add(row['document'])
            count += 1
            if count == 3:
                break
    corpus = packet['statistics']['product']
    measured = [summarized(row) for sign in (1, -1) for row in
                sorted(search_rows, key=lambda row: -sign * row['selected']['product'])[:8]]
    return client.request('control-plan', prompts.CONTROL,
        {'explanation': explanation, 'corpus_extremes': {key: corpus[key] for key in ('min', 'max')},
         'natural_examples': natural, 'measured_search': measured}, contracts.PLAN, 3,
        check=lambda output: check(output, tokenizer))


def run(client, measure, explanation, packet, search_rows, root, timeline, plan=None):
    tokenizer = measure.runtime.tokenizer
    if plan is not None:
        try:
            check(plan, tokenizer)
        except ValueError:
            plan = None
    if plan is None:
        plan = design(client, explanation, packet, search_rows, tokenizer)
    corpus = packet['statistics']['product']
    timeline.status('controlled-tests')
    original_rows, original_figures = measure.controls(plan['controls'])
    write(root / 'measurements/figures-initial.json', original_figures)
    observed = feedback(original_rows, corpus)
    write(root / 'controls-feedback.json', observed)
    timeline.status('controlled-test-refinement')
    refined = client.request('control-refinement', prompts.CONTROL_REFINE,
        {'explanation': explanation, 'previous_plan': plan, **observed}, contracts.PLAN, 3,
        check=lambda output: check(output, tokenizer))
    timeline.status('controlled-tests-revised')
    revised_rows, figures = measure.controls(refined['controls'], round_name='refined')
    write(root / 'controls-feedback-refined.json', feedback(revised_rows, corpus))
    selected = choose_versions(original_figures, figures, corpus, explanation)
    write(root / 'measurements/figures-revised.json', figures)
    write(root / 'measurements/figures.json', selected)
    return refined, original_rows + revised_rows, selected
