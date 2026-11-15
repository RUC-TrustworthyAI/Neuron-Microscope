"""Declare questions, templates and competing predictions before measuring."""
from .contracts import arr, obj, string

TEXT = {'type': 'string', 'minLength': 1}
SHORT = {'type': 'string', 'minLength': 1, 'maxLength': 48}
ROUNDS = 2
MAX_INVESTIGATIONS = 5

PLAN_PROMPT = """Plan a neuron investigation from the original Criticism. Return only JSON.
You have at most two rounds, each with at most five investigations; zero is allowed.
Use original_criticism, the frozen interpretation, ALL previous_rounds (including
failures and inconclusive outcomes), and working_interpretation together. Round two
must revisit the original concerns as well as attack proposed first-round revisions.
Each question needs a plausible rival and an observation that distinguishes them.
Do not repeat answered questions or rephrase earlier tests. If no new distinguishable
question remains, decision=stop and investigations=[]. Never fill a quota.
Choose review for supplied saved evidence, contrast for controlled text comparisons,
or corpus for natural examples from the complete saved activation archive. Link each
question to original criticism IDs. Use review when existing evidence is sufficient.
Evidence previews are navigation aids, not full causal prefixes.
Case outcomes retain every observation with bounded input excerpts; full saved
contexts can be inspected next round with a review of the supplied case IDs.
Corpus queries are exploratory literal retrieval, not a representative semantic benchmark. Context and
lexical/syntactic rivals matter; high activation alone does not establish specificity
or mechanism. Predeclare predictions under BOTH explanations before observing results.
Only text fields and listed IDs are yours to supply; measurements come from Python.
"""

SPEC_PROMPT = """Return only the prescribed JSON experiment specification.
The question and competing predictions have already been registered. Preserve them.
For contrast: give literal prefix and suffix frames and ONLY the replacement fragments
in conditions; Python concatenates them. No {slot}, markers, full-sentence extraction,
regex, code, or token indices. Conditions apply to every frame. Use multiple frames
when a claim should generalize across contexts, and change only the disputed factor.
The first condition is the reference; each condition declares predictions under both
explanations. Choose replacement_span, after_replacement, or last_token. Scoring is
signed absmax on the declared span, or single on the last token; it excludes common
prefix tokens. after_replacement requires a nonempty suffix; replacement_span requires
nonempty replacements. Respect the supplied token budget; inputs are never truncated.
For corpus: specify two to four labeled groups of literal terms (OR within each group).
The program scans the full archive, excludes initial interpretation documents, samples
matching documents uniformly by fixed seed without seeing activations, and scores the
first matched span. Use genuinely distinguishing groups; no regex or activation filter.
"""

ASSESS_PROMPT = """Return only JSON assessing this preregistered question.
Compare ALL supplied results with BOTH predictions. Cite only supplied evidence IDs.
Use survived, narrowed, rejected, or inconclusive for the working hypothesis. Empty
retrieval, failed measurements, overlapping query groups, or indistinguishable
predictions cannot establish survival. Corpus matching is exploratory and lexical;
do not claim semantic coverage or unbiased prevalence. Use signed native product,
not gate alone. Contrasts exclude common prefix tokens. Report real magnitudes only
from measurements; do not invent statistics.
Only causal_prefix (through the measured anchor) can explain that anchor's activation;
text after causal_end_char is linguistic context, never a cause of the earlier response.
Input excerpts carry original character offsets and can omit earlier context; do not
infer the absence of a contextual cue from an excerpt. Request a review if full context
is needed for a verdict. Every case's measured outcome is supplied, not just successes.
Evidence reused from the baseline and adaptively generated follow-ups are not held-out validation. RelP is attribution,
not causal proof. A suggested_revision is a proposed refinement for the next round,
not a replacement for the frozen original explanation. State unresolved limits.
"""


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
