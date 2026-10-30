"""The model supplies text and references; measured values stay in Python."""


def obj(**fields):
    return {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}


def arr(item, minimum=0, maximum=None):
    result = {'type': 'array', 'items': item, 'minItems': minimum}
    if maximum is not None:
        result['maxItems'] = maximum
    return result


def string(*choices):
    result = {'type': 'string'}
    if choices:
        result['enum'] = list(choices)
    return result


S = string()
POLE = string('positive', 'negative')
RULE = obj(id=S, name=S, description=S, predicted_pole=POLE)
CLAIM = obj(title=S, lede=S, one_paragraph=S, negative_pole=S, what_would_change_my_mind=S,
            confidence=string('low', 'medium', 'high'))
def interpretation_schema(max_rules_per_pole=3):
    return obj(claim=CLAIM, response_rules=arr(RULE, 2, 2 * max_rules_per_pole),
               hypotheses=arr(obj(id=S, description=S, rival=S), 2, 6))


INTERPRETATION = interpretation_schema()
SHORT_LABEL = {'type': 'string', 'minLength': 1, 'maxLength': 48,
               'description': 'A condition name: preferably one word, at most four words. The full input belongs in text.'}
CANDIDATE = obj(label=SHORT_LABEL, text=S)
REGION = obj(id=S, name=S, pole=POLE, rationale=S, candidates=arr(CANDIDATE, 1))
SEARCH = obj(regions=arr(REGION, 2, 8))
CONTROL = obj(id=S, title=S, rationale=S, template=S,
              score=string('replacement_span', 'after_replacement'),
              reduction=string('absmax', 'single'),
              replacements=arr(obj(label=SHORT_LABEL, text=S, expected=string('positive', 'negative', 'neutral')), 10, 10))
PLAN = obj(controls=arr(CONTROL, 2, 4), family_sentences=arr(S, 3, 5))
PREDICTIONS = obj(predictions=arr(obj(id=S, group=S, predicted_pole=string('positive', 'negative', 'uncertain'), reason=S)))
BLIND = obj(positive=arr(CANDIDATE, 6, 6), negative=arr(CANDIDATE, 6, 6))
REVISION = obj(claim=CLAIM, response_rules=arr(RULE, 2, 6),
    criticism=arr(obj(attack=S, prediction=S, result=S,
                     verdict=string('survived', 'narrowed', 'rejected', 'inconclusive'), evidence_ids=arr(S)), 3, 5),
    section_prose=obj(search=S, spectrum=S, anatomy=S, family=S, circuit=S, ablation=S, tests=S),
    clusters=arr(obj(id=S, name=S, story=S, graph_ids=arr(S), example_graph_ids=arr(S, 0, 3))),
    units=arr(obj(id=S, label=S)),
    graph_captions=arr(obj(id=S, title=S, why=S)))
REPORT_ANALYSIS = obj(**{key: value for key, value in REVISION['properties'].items()
                        if key not in ('claim', 'response_rules')})
REVIEW = obj(group=string('one_idea', 'branching', 'unplaced', 'two_related', 'mixture', 'unresolved', 'none'),
             organized={'type': 'integer', 'minimum': 1, 'maximum': 5},
             understood={'type': 'integer', 'minimum': 1, 'maximum': 5},
             one_line=S, main_job=S,
             where_it_branches=arr(obj(branch=S, grows_from=S, path=S, size=S, evidence=S)),
             remainders=arr(obj(what=S, evidence=S)), concerning=arr(S), fine=arr(S))


def validate(value, schema, path='$'):
    kind = schema['type']
    valid = {'object': isinstance(value, dict), 'array': isinstance(value, list),
             'string': isinstance(value, str), 'integer': type(value) is int}[kind]
    if not valid:
        raise ValueError(f'{path}: expected {kind}')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{path}: unexpected value {value!r}')
    if kind == 'object':
        missing = set(schema['required']) - value.keys()
        extra = value.keys() - schema['properties'].keys()
        if missing or (extra and not schema.get('additionalProperties', True)):
            raise ValueError(f'{path}: missing={sorted(missing)}, extra={sorted(extra)}')
        for key, item in value.items():
            if key in schema['properties']:
                validate(item, schema['properties'][key], path + '.' + key)
    elif kind == 'array':
        if len(value) < schema.get('minItems', 0) or len(value) > schema.get('maxItems', float('inf')):
            raise ValueError(f'{path}: unexpected item count {len(value)}')
        for i, item in enumerate(value):
            validate(item, schema['items'], f'{path}[{i}]')
    elif kind == 'string':
        if not schema.get('minLength', 0) <= len(value) <= schema.get('maxLength', float('inf')):
            raise ValueError(f'{path}: unexpected text length')
    elif kind == 'integer':
        if not schema.get('minimum', value) <= value <= schema.get('maximum', value):
            raise ValueError(f'{path}: value out of range')


def validate_rules(output, max_rules_per_pole=3):
    rules = output['response_rules']
    if len({r['id'] for r in rules}) != len(rules) or any(r['id'] == 'no-visible-cue' for r in rules):
        raise ValueError('Directional rule IDs must be unique and exclude no-visible-cue')
    for pole in ('positive', 'negative'):
        if sum(r['predicted_pole'] == pole for r in rules) > max_rules_per_pole:
            raise ValueError(f'At most {max_rules_per_pole} rules per pole')


def control_request(plan):
    """Translate position semantics explicitly; no marker search or implicit final token."""
    template = plan['template']
    if template.count('{slot}') != 1:
        raise ValueError('A controlled experiment needs exactly one {slot}')
    if any(len(r['label'].split()) > 4 or len(r['label']) > 48 for r in plan['replacements']):
        raise ValueError('Use one word or a short condition label of at most four words and 48 characters')
    prefix, suffix = template.split('{slot}')
    replacements = []
    for row in plan['replacements']:
        text = row['text']
        if not text.startswith(prefix) or not text.endswith(suffix) or len(text) < len(prefix) + len(suffix):
            raise ValueError('Condition text must be the complete input with the exact template prefix and suffix')
        stop = len(text) - len(suffix) if suffix else len(text)
        replacements.append({**row, 'text': text[len(prefix):stop]})
    mode = plan['score']
    if mode == 'after_replacement' and not suffix.strip():
        raise ValueError('after_replacement needs a nonempty scored suffix')
    return {'template': template, 'replacements': replacements,
            'score': {'location': mode, 'reduction': plan['reduction']}}
