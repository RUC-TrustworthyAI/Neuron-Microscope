---
name: experiment
description: Design and measure neuron-paper contrasts, RelP graphs, routes and independent tests.
disable-model-invocation: true
metadata:
  disco-role: operating
---

# Experiments

Read `/inputs/handoff/handoff.json` and its report for the initial explanation. Use the evidence tool to inspect saved cases. This stage plans the full experiment group: Activation search, Clusters, Inside the neuron, Where it sits, Circuit and Tests.

Read the report by section with scripts/read_evidence.py. The API reference below gives the complete input formats. Read structured results and the fields needed for the analysis; tool formats do not need to be inferred from HTML or filesystem searches. Complete measurements remain in their referenced JSON files.

## References

- [API reference](references/api-reference.md): one experiment plan, measurements and blind tests.
- [Workflow](references/workflows.md): evidence, measurement, naming and handoff.

## Work

Read the evidence index first. In an existing-only run, follow the Existing-data runs section of the workflow: read CPU coactivation, reuse measurements, label the activation spectrum and semantic groups, prepare unmeasured text tests, and complete the report without new inference.

Define the full set of semantic groups from the saved positive, negative and intermediate contexts, then call scripts/label_spectrum.py with all groups together. Supply id, name, description and pole for each group. A fresh call labels every unique saved report position, including extrema, quantile examples and scatter positions; a single group or case_ids inside a group does not make a small pilot. Use the complete JSON example in the API reference. Repeat an unchanged groups request only to retrieve its saved result; use resume_from to label missing cases with the same definitions. Changing the definitions starts a new full classification. Use its labels and counts for Activation spectrum and Inside the neuron. Give each semantic group an explicit response pole: positive, negative or either. Use either for mixed, background or unclear groups. These describe the group's semantic response, not every member's sign. The labelling tool fills missing poles from saved evidence; plots use those explicit annotations. The report displays the main activation distribution. Unmeasured text tests remain internal experiment plans.

Choose representative positive, negative and intermediate cases. Submit one measurement group with:
- Plan replacement sweeps separately for the observed positive and negative tails. Include at least one controlled sweep for each observed pole; use the remaining sweep to distinguish the strongest alternative. Mark each sweep pole as positive, negative or both. A weaker positive control is not a negative-pole experiment. For a mixed or unclear tail, test its leading explanation against matched alternatives and keep that uncertainty visible. Negative experiments normally use min scoring; positive experiments normally use max scoring. Choose a common scored span or anchor before measuring, and retain the relevant observed prefix. Do not invent an unobserved second pole. Reuse completed sweeps when continuing a run; add only an uncovered pole.
- Illustrated RelP graphs covering observed positive and negative neuron roots, and a predicted output-logit root where useful. Use the configured attribution node and edge counts.
- Search candidates and matched controls for both observed poles.
- Saved anatomy examples and a joint family comparison.

Read /control/settings.json for the experiment and sampling parameters. The host selects the natural atlas, preserves original token IDs from document start through the anchor, and adds representative cases from each semantic group. Plan the configured rounds: initial tests, then strong counterexamples and matched alternatives.

Every candidate, control and replacement declares its scored token or span and reduction before measurement. Distinguish the changed cue from the response location. Use the same rule for matched conditions. Existing natural cases retain their saved anchor. There is no implicit maximum over a replacement or a full sentence.

The host automatically registers every new completed condition, measures target zeroing with same-layer random controls, and computes background local RelP. Same input/anchor/settings reuse their recorded measurements. These follow-ups feed What it does and the measured route data. Natural examples define the atlas; probe assignments remain identified as probes. Choose representative multi-hop display graphs separately.

Blind maximum and Blind minimum are a fixed final measurement step of this stage. The host runs the isolated writer at completion unless matching complete results already exist. You may call scripts/run_tests.py earlier to inspect them; completion reuses that result. The writer sees only the frozen initial explanation, writes the configured positive and negative candidates, and the host measures their full-input maximum and minimum before comparing them with the saved corpus extremes. Route prediction is paused; leave that report slot as —.

At completion, the host labels every newly measured search, control and replacement anchor from its observed prefix and native product. Saved assignments are reused; only new anchors need semantic classification. The host also connects semantic groups to the scatter clusters; colours inherit from the first coloured member group, as in the report renderer. This uses saved measurements and does not run the target model.

Complete with a result summary and annotations using the full JSON example in the API reference. Each annotations.routes value is an object with name and description; each annotations.units value is a label string; section prose uses an object with summary. Use measured route and unit IDs from the results. Completion returns the experimental report with the incoming claim unchanged. Submit after the measurements, spectrum labels and independent tests; this task does not revise the claim.

## Semantic Clusters and raw routes

A route is a numerical contribution pattern, not a semantic category. Keep its measured membership intact in Raw factorization. After the configured measurement rounds and scripts/run_tests.py have returned, read scripts/read_evidence.py --section routes. This gives the complete current natural-training member index. Use --section routes --ids GRAPH_ID ... to read complete observed prefixes, native traces, selected anchors and upstream vectors. The index previews are navigation aids, not sufficient evidence for deciding a member's meaning. Read every member's complete context before assigning it. Read the latest index again if you add experiments after it.

Organize those measured members into reader-facing semantic clusters. Several routes may contribute to one cluster; one route may contain members belonging to different clusters. Use the text, scored position, activation and upstream contributions together. An existing spectrum label is a hint, not a compulsory assignment. Similar text does not prove the same mechanism; similar contributions do not prove the same topic.

Give every member one display cluster, without dropping weak activations, mismatches or counterexamples. Use mixed or unresolved clusters when no supported common account exists. Explain when a numerical route crosses semantic categories. If splitting out a member limits a proposed explanation, retain that limitation explicitly in the affected story; do not use regrouping to make contrary evidence disappear.

For each cluster, state its membership criterion, the shared observations or contribution patterns, and the important exceptions or uncertainty. A name inferred from a few strong examples must not silently describe all remaining members. Supply positive, negative or either as the semantic response pole; use either for mixed or unresolved groups.

Choose up to three example_graph_ids from the cluster's actual graph_ids. Show a typical case and, whenever one is informative, a boundary case or counterexample. Do not merely choose the first or highest-activation members. The complete member list remains in the structured report.

Return annotations.clusters as an array of {id, name, story, pole, graph_ids, example_graph_ids}. Use graph_id values from the routes evidence, not saved case IDs or route IDs. The fixed tools derive member_routes, graph counts, attribution-mass shares, upstream summaries and original example values. Do not supply those numbers. Raw routes retain their own annotations.routes name/description mapping.

## Writing

Write a short title in the form “[concept] neuron”, with an optional phrase describing its response. Begin the explanation directly with what this neuron responds to or changes. Refer to the measured unit as “this neuron”; keep its layer and index in the identification fields.

Use ordinary descriptions of the positive and negative responses. Describe sparsity in the activation-distribution discussion, rather than calling the neuron a “sparse feature”. Keep qualifications in sentences instead of stacking them in a long title.

Describe the observations and conclusions directly for the reader.
