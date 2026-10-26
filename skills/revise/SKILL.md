---
name: revise
description: Revise the neuron explanation, six analysis sections, figure captions and final Criticism from measured results.
disable-model-invocation: true
metadata:
  disco-role: operating
---

# Report

Read `/inputs/handoff/handoff.json`, its HTML and measurements. Read the condition registry for the actual full-prefix anchors, scoring definitions, ablations and contribution vectors. Use measured denominators from those records. Revise the explanation using the search, route clusters, gate/value anatomy, family, RelP circuit measurements and the semantic activation spectrum together.

Read the handoff as an overview: measurement_summary contains experiment status, key statistics and paths; complete measurements remain in the referenced files. Use scripts/read_evidence.py to read saved evidence by section or case, and use --section measurements to list experiments or --section measurements --ids EXPERIMENT_ID to read one experiment overview and its file names. For details, load the referenced JSON file in Python and print the fields needed for the question. Full texts, traces, contribution vectors and plotting arrays remain available in those files. The host renders the HTML from your submitted interpretation and annotations.

## References

- [API reference](references/api-reference.md): claim, Criticism and annotations.
- [Workflow](references/workflows.md): report submission.

## Write

For existing-data runs, read the saved coactivation and proposed text tests in the measurements. Describe neighbour associations using their sampled correlation and same-input traces. Semantic Clusters are groups of selected contexts. Neither supplies causal routes or upstream/downstream contributions. Proposed texts are internal experiment plans with no measured activation score; they are not publication sections. Discuss counterexamples and select a representative saved strip when helpful. The fixed tools fill direct output weights and numerical group counts.

Explain which observations support each pole and which suggest an alternative. Name the measured routes, describe representative gate/value traces, and annotate the figures and six analysis sections. Distinguish observed local RelP contributions, native neuron-zeroing effects and corpus associations. Circuit tables aggregate the natural atlas with downstream coverage limited to nominated same-position readers. New probes have separate attribution records and route assignments. Use their zeroing results and matched controls when revising the interpretation. Use the measured independent texts in Criticism, including failures to reach the corpus extreme. Route prediction is paused; leave it as — and make no claim about its accuracy. Discuss semantic-group counts as counts of the selected samples. Keep the initial-version blind tests attached to the explanation they tested.

Submit through `scripts/write_report.py`. Numerical fields are filled from the saved measurements. Default language is English; unavailable fields remain —. This stage runs no new experiments.

## Review semantic Clusters

Raw NMF routes are numerical contribution patterns. The experiment handoff separately stores cluster_definitions: semantic cluster names, stories, graph_ids and example_graph_ids. Read scripts/read_evidence.py --section routes for the complete membership index and submitted definitions; use --ids GRAPH_ID ... for full member contexts and attribution vectors. Read the members supporting and challenging each story, including weak or mixed cases. Do not treat a route ID or its name as a semantic verdict.

A semantic cluster can combine members of several routes, and a raw route can contribute to several semantic clusters. Keep every measured member represented, retain counterexamples and uncertainty, and explain any semantic split within a raw route. Do not remove inconvenient members or narrow the displayed examples to conceal a failed generalization. Prefer a typical example plus any informative boundary or contrary case within the existing three-example display.

If the definitions remain supported, omit annotations.clusters: the existing memberships and examples are preserved even when routes or unit labels are edited. To revise semantic groups or their prose, submit the complete clusters array with id, name, story, pole, graph_ids and example_graph_ids. The host recomputes numerical summaries from measured members. Raw route membership never changes. This review uses saved measurements only.

## Writing

Write a short title in the form “[concept] neuron”, with an optional phrase describing its response. Begin the explanation directly with what this neuron responds to or changes. Refer to the measured unit as “this neuron”; keep its layer and index in the identification fields.

Use ordinary descriptions of the positive and negative responses. Describe sparsity in the activation-distribution discussion, rather than calling the neuron a “sparse feature”. Keep qualifications in sentences instead of stacking them in a long title.

Describe the observations and conclusions directly for the reader.

Place direct output weights (Promotes/Suppresses) and neuron-ablation findings in annotations.sections.ablation (What it does). Use annotations.sections.circuit for upstream writers and downstream readers. The renderer places the numerical tables in those same sections.

## Measured quantities

Circuit tables store signed local RelP contributions: source activation times the modified derivative of the reader activation, with intervening MLP outputs detached. Upstream contributions are pooled over source positions; downstream measurements cover the nominated readers at the anchor position. The illustrated multi-hop graph additionally divides each local contribution by the receiver activation and multiplies by that receiver's influence on the graph root. These two displayed weight quantities are distinct. Route factorization uses sign-split, L1-normalized local contribution vectors.
