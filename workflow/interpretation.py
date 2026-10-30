"""Explain saved spectrum measurements, then refine once against blind predictions."""
from . import contracts, prompts
from .evidence import prediction_feedback
from .labels import predict
from .state import write


def explain(client, packet, sample, contexts, root, timeline, max_rules_per_pole=3):
    schema = contracts.interpretation_schema(max_rules_per_pole)
    def check(output):
        contracts.validate_rules(output, max_rules_per_pole)
    limit = 'three' if max_rules_per_pole == 3 else str(max_rules_per_pole)
    timeline.status('initial-explanation')
    initial = client.request('initial', prompts.INTERPRET.format(max_rules_per_pole=limit),
                             packet, schema, 2, check=check)
    timeline.status('initial-blind-labels')
    before = predict(client, initial, sample, contexts, root / 'initial-labels',
                     name='initial-labels', stage=2)
    timeline.status('explanation-refinement')
    feedback = prediction_feedback(before)
    feedback['assignments'] = [{key: row[key] for key in
        ('id', 'group', 'predicted_pole', 'reason', 'activation', 'true_pole', 'correct')}
        for row in before['records']]
    # The complete measured contexts are already in packet; avoid repeating their text in feedback.
    for group in feedback['groups']:
        group['examples'] = [{key: value for key, value in row.items()
                              if key not in ('observed_prefix', 'nearby_continuation')}
                             for row in group['examples']]
    refined = client.request('refinement', prompts.REFINE.format(max_rules_per_pole=limit),
        {**packet, 'current_explanation': initial, 'prediction_feedback': feedback},
        schema, 2, check=check)
    timeline.status('refined-blind-labels')
    after = predict(client, refined, sample, contexts, root / 'refined-labels',
                    name='refined-labels', stage=2)
    write(root / 'interpretation-history.json', [
        {'round': 0, 'version': before['version'], 'summary': before['summary']},
        {'round': 1, 'version': after['version'], 'summary': after['summary']}])
    write(root / 'interpretation.json', refined)
    write(root / 'predictions.json', after)
    return initial, refined, before, after
