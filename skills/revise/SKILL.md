---
name: revise
description: Revise the neuron explanation, six analysis sections, figure captions and final Criticism from measured results.
disable-model-invocation: true
metadata:
  disco-role: operating
---

# Report

Start with `python scripts/read_evidence.py --section brief`. It returns the incoming explanation, section summaries, experiment index, saved-file paths and prediction-versus-measurement cases together. Use these to locate the evidence for revision. Read the condition registry for the actual full-prefix anchors, scoring definitions, ablations and contribution vectors. Revise the explanation using the search, route clusters, gate/value anatomy, family, RelP circuit measurements and the semantic activation spectrum together.

Use `--sections SECTION ...` to read several complete evidence sections in one call and `--ids ID ...` for several complete cases. The brief's experiment_index contains status, statistics, directory paths and JSON file names. Read the specific contexts and measured fields needed for the claim using those IDs and paths; full texts, traces, contribution vectors and plotting arrays remain available. The host renders the HTML from the submitted interpretation and annotations.

## References

- [API reference](references/api-reference.md): claim, Criticism and annotations.
- [Workflow](references/workflows.md): report submission.

## Write

For existing-data runs, read the saved coactivation and proposed text tests in the measurements. Describe neighbour associations using their sampled correlation and same-input traces. Semantic Clusters are groups of selected contexts. Neither supplies causal routes or upstream/downstream contributions. Proposed texts are internal experiment plans with no measured activation score; they are not publication sections. Discuss counterexamples and select a representative saved strip when helpful. The fixed tools fill direct output weights and numerical group counts.

Explain which observations support each pole and which suggest an alternative. Name the measured routes, describe representative gate/value traces, and annotate the figures and six analysis sections. Distinguish observed local RelP contributions, native neuron-zeroing effects and corpus associations. Circuit tables aggregate the natural atlas with downstream coverage limited to nominated same-position readers. New probes have separate attribution records and route assignments. Use their zeroing results and matched controls when revising the interpretation. Use the measured independent texts in Criticism, including failures to reach the corpus extreme. Route prediction is disabled in this pipeline; leave it as — and make no claim about its accuracy. Discuss semantic-group counts as counts of the selected samples. Keep the initial-version blind tests attached to the explanation they tested.

Match each figure's conclusion to its actual selected token or scored span, reading the saved native trace where needed. A flat response at one token does not imply inactivity across the sentence or refute a hypothesis about another position. Distinguish cue tokens, the boundary before an upcoming value and tokens within or after that value. Captions name the measured location and quantity; product is not raw gate. Preserve failed or uninformative tests and explain their scope. Prefer informative controlled contrasts and strong counterexamples for the main figures; keep flat diagnostic tests available without presenting magnified near-zero residuals as a strong effect.

Select exactly three representative measured graphs for Three graphs, using annotations.graph_ids in display order. Cover both observed poles and a distinct mechanism or informative counterexample. Additional measured graphs remain in the saved artifacts. Preserve the ten-condition common-frame headline selected with annotations.headline_figure_id; change that ID only to another measured ten-condition comparison covering the same evidence needs. Keep actual failed controls and measured signs. The ten-condition figures and eleven short family inputs are measured during the experiment stage; this stage supplies their explanation and captions.

Finish wording and evidence edits in the local submission JSON, then submit the final explanation through `scripts/write_report.py`. Submission runs the independent review and response predictions; it is a final submission tool. Wait for its report path and stop. For a subsequent update limited to captions, section introductions or displayed figure selections, use `scripts/render_report.py`; that tool preserves the submitted claim, response rules and predictions and performs only rendering. Numerical fields come from saved measurements. Default language is English; unavailable fields remain —. This stage runs no new experiments.

For each Criticism row choose exactly one verdict code: `survived` (the tested claim survived as written), `narrowed` (it needs qualification or revision), or `rejected` (it failed). Write the reasoning in `result` and the challenge in `attack`; the verdict field is only the short badge code.

## Review semantic Clusters

Raw NMF routes are numerical contribution patterns. The experiment handoff separately stores cluster_definitions: semantic cluster names, stories, graph_ids and example_graph_ids. Read scripts/read_evidence.py --section routes for the complete membership index and submitted definitions; use --ids GRAPH_ID ... for full member contexts and attribution vectors. Read the members supporting and challenging each story, including weak or mixed cases. Do not treat a route ID or its name as a semantic verdict.

A semantic cluster can combine members of several routes, and a raw route can contribute to several semantic clusters. Keep every measured member represented, retain counterexamples and uncertainty, and explain any semantic split within a raw route. Do not remove inconvenient members or narrow the displayed examples to conceal a failed generalization. Prefer a typical example plus any informative boundary or contrary case within the existing three-example display.

Keep spectrum and scatter buckets compact: at most three positive groups, at most three negative groups, and one No visible cue group for cases whose direction cannot be distinguished. This applies to response_rules. For measured route-based Clusters, preserve genuinely distinct recurring patterns and every counterexample; do not promote incidental topics or individual entities into separate neuron meanings.

For Inside the neuron, coordinates are measured raw gate and up; point colours show independent predictions from the frozen explanation. Read scripts/read_evidence.py --section response_predictions to inspect the initial per-sample labels, coverage, prediction-versus-measurement counts and full-prefix mistakes. These labels were produced immediately after Agent 1 submitted its initial explanation and buckets; use them alongside the experiment results when revising the account. Positive and negative truth are determined by measured product at the anchor; the predictor sees only the fixed explanation, its response_rules, text and anchor position. Distinguish feedback samples, already-seen evidence and experiments. Examine recurring strong mistakes and uncertain predictions before revising the claim; near-zero sign disagreements are weak evidence. Preserve counterexamples. Explain consequential mismatches and any resulting qualification in Criticism, using the actual feedback records.

Submit response_rules alongside the revised claim, retaining or revising the initial rules as evidence requires. Use at most three positive groups and at most three negative groups. Include exactly one uncertain group with id no-visible-cue, name No visible cue and predicted_pole uncertain. Each group has id, name, description and predicted_pole; define textual conditions and anchor locations. Assign all cases without a distinguishable direction to No visible cue. Activation spectrum and Inside the neuron retain the shared samples and predictions from the initial explanation. Use that feedback when revising, and keep its explanation version explicit. The measured bins and coordinates remain fixed; opposite-side colours expose disagreements with the initial rules.

If the definitions remain supported, omit annotations.clusters: the existing memberships and examples are preserved even when routes or unit labels are edited. To revise semantic groups or their prose, submit the complete clusters array with id, name, story, graph_ids and example_graph_ids. The host recomputes numerical summaries from measured members. Raw route membership never changes. This review uses saved measurements only.

## Writing

Write a short title in the form “[concept] neuron”, with an optional phrase describing its response. Begin the explanation directly with what this neuron responds to or changes. Refer to the measured unit as “this neuron”; keep its layer and index in the identification fields.

Use ordinary descriptions of the positive and negative responses. Describe sparsity in the activation-distribution discussion, rather than calling the neuron a “sparse feature”. Keep qualifications in sentences instead of stacking them in a long title.

Describe the observations and conclusions directly for the reader.

Use approximate English word targets: lede 60–85; one_paragraph 150–180; negative_pole 80–120; what_would_change_my_mind 40–60. The latter three fields form The claim in detail. Activation spectrum summary is 35–55 words; Activation search summary is 45–70 words. Circuit introduction is approximately 30 words. Other section introductions are 25–50 words; cluster stories and figure captions are 35–65 words. Give the interpretation and decisive evidence rather than narrating experiment rounds or enumerating every result. Use short noun phrases for unit and group labels. Preserve contrary or unresolved cases in evidence, captions and Criticism results. These are writing targets, not truncation or acceptance rules.

Place direct output weights (Promotes/Suppresses) and neuron-ablation findings in annotations.sections.ablation (What it does). Use annotations.sections.circuit for upstream writers and downstream readers. The renderer places the numerical tables in those same sections.

## Measured quantities

Circuit tables store signed local RelP contributions: source activation times the modified derivative of the reader activation, with intervening MLP outputs detached. Upstream contributions are pooled over source positions; downstream measurements cover the nominated readers at the anchor position. The illustrated multi-hop graph additionally divides each local contribution by the receiver activation and multiplies by that receiver's influence on the graph root. These two displayed weight quantities are distinct. Route factorization uses sign-split, L1-normalized local contribution vectors.
