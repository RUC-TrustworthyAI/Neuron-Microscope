"""Declare questions, templates and competing predictions before measuring."""
from .contracts import arr, obj, string

TEXT = {'type': 'string', 'minLength': 1}
SHORT = {'type': 'string', 'minLength': 1, 'maxLength': 48}
ROUNDS = 2
MAX_INVESTIGATIONS = 5


def plan_schema(criticism_ids, evidence_ids):
    investigation = obj(question=TEXT, criticism_ids=arr(string(*criticism_ids), 1),
                        action=string('review', 'contrast', 'corpus'),
                        evidence_ids=arr(string(*evidence_ids), 0, 8),
                        hypothesis=TEXT, rival=TEXT, prediction_if_hypothesis=TEXT,
                        prediction_if_rival=TEXT, distinguishing_observation=TEXT, why_new=TEXT)
    return obj(decision=string('continue', 'stop'), reason=TEXT,
               investigations=arr(investigation, 0, MAX_INVESTIGATIONS))


CONTRAST = obj(score_location=string('replacement_span', 'after_replacement', 'last_token'),
    frames=arr(obj(prefix=string(), suffix=string()), 1, 3),
    conditions=arr(obj(label=SHORT, replacement=string(),
                       prediction_if_hypothesis=TEXT, prediction_if_rival=TEXT), 2, 6))
CORPUS = obj(groups=arr(obj(label=SHORT, terms=arr(TEXT, 1, 8)), 2, 4))
SPECIFICATIONS = {'contrast': CONTRAST, 'corpus': CORPUS}


def assessment_schema(evidence_ids):
    return obj(result=TEXT, verdict=string('survived', 'narrowed', 'rejected', 'inconclusive'),
               suggested_revision=string(), evidence_ids=arr(string(*evidence_ids)))


def check_plan(plan, evidence_ids):
    if (plan['decision'] == 'continue') != bool(plan['investigations']):
        raise ValueError('continue needs investigations; stop needs an empty list')
    for item in plan['investigations']:
        # Nonempty ID lists are enforced by the schema's enum constraints.
        if not evidence_ids and item['evidence_ids']:
            raise ValueError('No saved evidence IDs are available')
        if item['action'] == 'review' and not item['evidence_ids']:
            raise ValueError('Review requires saved evidence IDs')
