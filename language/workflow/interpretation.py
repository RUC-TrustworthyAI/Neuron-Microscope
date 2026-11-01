"""Explain saved spectrum measurements, then refine against blind predictions."""
from . import contracts, prompts
from .evidence import refinement_feedback
from .labels import predict
from .state import write


def prediction_scores(predictions, sample):
    """Score signed predictions by magnitude, separately for each measured pole."""
    records = {row['id']: row['predicted_pole'] for row in predictions['records']}
    totals, weights = {}, {}
    for point in sample['points']:
        value = point['activation']
        if value == 0:
            continue
        pole = 'positive' if value > 0 else 'negative'
        predicted = records.get(point['id'], 'uncertain')
        agreement = 0 if predicted == 'uncertain' else 1 if predicted == pole else -1
        totals[pole] = totals.get(pole, 0) + abs(value) * agreement
        weights[pole] = weights.get(pole, 0) + abs(value)
    return {pole: totals[pole] / weight for pole, weight in weights.items()}


def select_round(history):
    """Prefer coverage above 90% accuracy, otherwise prefer accuracy."""
    eligible = [entry for entry in history if entry['summary']['directional_accuracy'] > .9]
    primary, secondary = ('directional_coverage', 'directional_accuracy') if eligible else (
        'directional_accuracy', 'directional_coverage')
    return max(eligible or history, key=lambda entry: (
        entry['summary'][primary], entry['summary'][secondary], entry['round']))


def save_current(root, explanation, predictions, history, selected_round):
    """Save one explanation together with the labels produced from it."""
    for entry in history:
        entry['selected'] = entry['round'] == selected_round
    write(root / 'interpretation-history.json', history)
    write(root / 'interpretation.json', explanation)
    write(root / 'predictions.json', predictions)




def explain(client, packet, sample, contexts, root, timeline, max_rules_per_pole=3, max_refinements=2):
    if type(max_refinements) is not int or max_refinements < 0:
        raise ValueError('max_refinements must be a nonnegative integer')
    schema = contracts.interpretation_schema(max_rules_per_pole)
    def check(output):
        contracts.validate_rules(contracts.interpretation_result(output), max_rules_per_pole)
    limit = 'three' if max_rules_per_pole == 3 else str(max_rules_per_pole)
    timeline.status('initial-explanation')
    initial = contracts.interpretation_result(client.request(
        'initial', prompts.INTERPRET.format(max_rules_per_pole=limit), packet, schema, 2, check=check))
    timeline.status('initial-blind-labels')
    before = predict(client, initial, sample, contexts, root / 'initial-labels',
                     name='initial-labels', stage=2)
    selected, predictions = initial, before
    history = [{'round': 0, 'version': before['version'], 'summary': before['summary'],
                'pole_scores': prediction_scores(before, sample),
                'explanation_path': 'api/initial/result.json',
                'predictions_path': 'initial-labels/predictions.json'}]
    pairs = {0: (initial, before)}
    save_current(root, selected, predictions, history, 0)
    current, current_predictions = initial, before
    for round_index in range(1, max_refinements + 1):
        suffix = '' if round_index == 1 else f'-{round_index}'
        phase, label_phase = 'refinement' + suffix, 'refined-labels' + suffix
        timeline.status('explanation-refinement', round=round_index, max_refinements=max_refinements)
        feedback = refinement_feedback(current_predictions, packet['statistics']['product'])
        current = contracts.interpretation_result(client.request(
            phase, prompts.REFINE.format(max_rules_per_pole=limit),
            {**packet, 'current_explanation': current, 'prediction_feedback': feedback},
            schema, 2, check=check))
        timeline.status('refined-blind-labels', round=round_index, max_refinements=max_refinements)
        current_predictions = predict(client, current, sample, contexts, root / label_phase,
                                      name=label_phase, stage=2)
        pairs[round_index] = current, current_predictions
        history.append({'round': round_index, 'version': current_predictions['version'],
                        'summary': current_predictions['summary'],
                        'pole_scores': prediction_scores(current_predictions, sample),
                        'explanation_path': f'api/{phase}/result.json',
                        'predictions_path': f'{label_phase}/predictions.json'})
        chosen = select_round(history)['round']
        selected, predictions = pairs[chosen]
        save_current(root, selected, predictions, history, chosen)
    return initial, selected, before, predictions
