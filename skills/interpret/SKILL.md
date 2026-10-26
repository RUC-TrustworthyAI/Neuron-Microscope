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

The target report is mounted read-only at `/inputs/report`. Run the bundled `scripts/read_evidence.py` to list evidence from that same report. Fetch the sections and contexts needed for the explanation. The anatomy_scan section supplies the amplitude-stratified corpus points and their semantic groups, prepared before this stage. Read these groups together with the original evidence when forming the initial explanation; inspect their observed prefixes as needed. Positive and negative extremes show the strongest responses; weaker quantile cases help establish the scope of the pattern.

Explain product activation. Gate and value support that explanation. Read each observed prefix together with its measured anchor. Following text is a prediction target. Token selectivity thresholds are fractions of the signed corpus extreme, not percentiles.

## Submit the initial explanation

Send the claim through `scripts/submit_interpretation.py`; see the [workflow](references/workflows.md). The tool saves the initial explanation and returns its report path. Stop after it returns.

## Writing

Write a short title in the form “[concept] neuron”, with an optional phrase describing its response. Begin the explanation directly with what this neuron responds to or changes. Refer to the measured unit as “this neuron”; keep its layer and index in the identification fields.

Use ordinary descriptions of the positive and negative responses. Describe sparsity in the activation-distribution discussion, rather than calling the neuron a “sparse feature”. Keep qualifications in sentences instead of stacking them in a long title.

Describe the observations and conclusions directly for the reader.
