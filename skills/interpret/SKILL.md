---
name: interpret
description: Read saved activation statistics, positive and negative contexts, quantile samples, token selectivity and measured output effects.
disable-model-invocation: true
metadata:
  disco-role: operating
---

# Activation Evidence

Form the initial explanation from `/inputs/report/index.html` and its structured measurement files. This container performs the first stage only.

Read the report by section with scripts/read_evidence.py. For HTML layout, read the markup around the section you need. The self-contained HTML and measurement JSON also embed millions of numeric values; load those files in Python and select named fields or cases instead of printing whole files into the conversation. All evidence remains available through the section and case queries.

## Read the focused references

- [API reference](references/api-reference.md): sections, sample groups and pagination.
- [Workflows](references/workflows.md): commands for statistics and natural contexts.

## Read the measurements

The target report is mounted read-only at `/inputs/report`. Run the bundled `scripts/read_evidence.py` to list evidence from that same report. Fetch the sections and contexts needed for the explanation. Positive and negative extremes show the strongest responses; weaker quantile cases help establish the scope of the pattern.

Explain product activation. A measured positive product is the positive response pole; a measured negative product is the negative response pole. The tools determine this sign. Semantic group descriptions are provisional interpretations of context and do not change it. Compare the positive and negative members, their magnitudes and prefixes before adopting a directional explanation. Small nonzero values have a sign but give weak evidence for an association. Revise broad definitions contradicted by recurring strong responses. Gate and value support that explanation. Read each observed prefix together with its measured anchor. Following text is a prediction target. Token selectivity thresholds are fractions of the signed corpus extreme, not percentiles.

## Submit the initial explanation

Alongside the initial claim, submit response_rules as the initial bucket labels. Define at most three positive groups and at most three negative groups, supported by the observed contexts. Fewer groups are appropriate when evidence supports fewer distinct patterns. Each group contains id, name, description and predicted_pole. Add exactly one uncertain group with id no-visible-cue, name No visible cue and predicted_pole uncertain. Put absent, ambiguous or conflicting cues in this single group whenever the response direction cannot be distinguished. Describe contextual conditions and the anchor position; do not define membership from the measured sign alone. These groups serve Activation spectrum and Inside the neuron. The host fixes this initial submission, immediately labels the individual samples from their complete observed prefixes and anchors, and saves predictions joined to measured responses. Those initial labels are passed onward as evidence for the revision agent.

Send the claim through `scripts/submit_interpretation.py`; see the [workflow](references/workflows.md). The tool saves the initial explanation, completes the initial per-sample labeling, and returns its report path. Stop after it returns.

## Writing

Write a short title in the form “[concept] neuron”, with an optional phrase describing its response. Begin the explanation directly with what this neuron responds to or changes. Refer to the measured unit as “this neuron”; keep its layer and index in the identification fields.

Use ordinary descriptions of the positive and negative responses. Describe sparsity in the activation-distribution discussion, rather than calling the neuron a “sparse feature”. Keep qualifications in sentences instead of stacking them in a long title.

Describe the observations and conclusions directly for the reader.

Use approximate English word targets: lede 60–85; one_paragraph 150–180; negative_pole 80–120; what_would_change_my_mind 40–60. The latter three fields are the three paragraphs of The claim in detail. State the main pattern, a few decisive measurements and the material limitation; keep exhaustive statistics and sample-by-sample discussion in evidence. These are writing targets, not output limits.
