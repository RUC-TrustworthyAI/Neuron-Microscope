---
name: experiment
description: Design and measure neuron-paper contrasts, RelP graphs, routes and independent tests.
disable-model-invocation: true
metadata:
  disco-role: operating
---

# Experiments

Start with `python scripts/read_evidence.py --section brief`. It returns the initial explanation, section summaries, experiment index, saved-file paths and prediction-versus-measurement cases together. This stage plans the full experiment group: Activation search, Clusters, Inside the neuron, Where it sits, Circuit and Tests.

Use `--sections SECTION ...` to read several complete evidence sections in one call and `--ids ID ...` to read several complete cases together. Inspect the contexts and measurements needed for each hypothesis using the returned IDs and paths. The API reference below gives the input formats. Full texts, traces and measurements remain available through the reader and referenced JSON files.

## References

- [API reference](references/api-reference.md): one experiment plan, measurements and blind tests.
- [Workflow](references/workflows.md): evidence, measurement, naming and handoff.

## Work

In an existing-only run, follow the Existing-data runs section of the workflow: read CPU coactivation, reuse measurements, label the activation spectrum and semantic groups, prepare unmeasured text tests, and complete the report without new inference.

Activation spectrum and Inside the neuron use the incoming explanation's frozen response_rules. The incoming buckets contain at most three positive groups, at most three negative groups, and one No visible cue group for indistinguishable directions. Agent 1's submission has already been followed by per-sample labeling. Read the existing independent predictions through scripts/read_evidence.py --section response_predictions. scripts/label_spectrum.py returns the saved rule groups and counts; submit {} to inspect them. Keep these definitions fixed during this stage. The independent labeler sees complete observed prefixes and anchor positions without measured activations, gates, up values or previous topic labels. The host bins spectrum samples by their actual activations and colours them by predicted rule membership. Opposite-side colours are prediction disagreements and remain visible.

Keep the incoming response buckets fixed: at most three positive, at most three negative, and one No visible cue. The spectrum and scatter reuse the same independent predictions for identical inputs. Preserve counterexamples and ambiguous cases for the revision stage. Put the three most informative figures first, covering observed poles and a meaningful rival or boundary case; additional experiments remain available in the report's expandable figures. Unit labels are short noun phrases; put experimental reasoning in the section explanation or caption.

When consolidating semantic groups or defining Inside the neuron scatter clusters, use contextual meanings that describe their members. Activation sign is an observed property of each measured token. It cannot be reassigned by a semantic label. A group with both signs needs an account of the differing contexts or an explicit unresolved explanation. Repeated strong disagreements require revisiting a definition; do not preserve a directional claim by calling all disagreements exceptions. Small nonzero responses retain their sign but provide weak evidence for a semantic response. Read the measured sign counts, ranges and concrete prefixes before drawing conclusions.

For English annotations, use approximately 35–55 words for the Activation spectrum introduction, 45–70 for Activation search, approximately 30 words for Circuit, and 25–50 for other section introductions. Cluster stories and figure captions are approximately 35–65 words. Describe the pattern and strongest qualification, with detailed measurements in the evidence. These are writing targets rather than response limits.

Choose representative positive, negative and intermediate cases. Submit one measurement group with:
- Plan replacement sweeps separately for the observed positive and negative tails. Include at least one controlled sweep for each observed pole; use the remaining sweep to distinguish the strongest alternative. Mark each sweep pole as positive, negative or both. A weaker positive control is not a negative-pole experiment. For a mixed or unclear tail, test its leading explanation against matched alternatives and keep that uncertainty visible. Negative experiments normally use min scoring; positive experiments normally use max scoring. Choose a common scored span or anchor before measuring, and retain the relevant observed prefix. Do not invent an unobserved second pole. Reuse completed sweeps when continuing a run; add only an uncovered pole.
- Three illustrated RelP graphs covering the observed positive and negative neuron roots plus a distinct mechanism, boundary case or output-logit root. Select exactly three representative measured graphs for publication; if extra graphs were needed during investigation, keep their data and submit only the three selected IDs in annotations.graph_ids. Use the configured attribution node and edge counts.
  Choose readable, representative inputs that expose the proposed cue and the scored response, rather than choosing only the largest corpus extrema. State which diagnostic cue actually occurs before the anchor; a following phone number or other continuation is not yet an input cause. An indirect entity association or background case is useful as a contrast, but should not be the only positive illustration. Retain the complete observed prefix for computation and distinguish first-token or generic-context contributions from semantic cues when describing the graph.
  An output-logit illustration should target a plausible next token supported by the measured baseline predictions. Do not automatically choose the concept's name as the output token. A deliberately unlikely output target is a contrast and must be described as such.
- Search candidates and matched controls for both observed poles.
- Saved anatomy examples and a joint family comparison using the short written inputs described below.

## Published comparisons and joint traces

Each replacement figure has exactly ten measured conditions in one fixed template, including the headline comparison. For the headline, choose candidates spanning likely positive responses, likely negative responses and unrelated controls from the current evidence. A useful design is four positive candidates, four negative candidates and two unrelated controls. These are experimental expectations; retain the measured signs and failures. Use the same response anchor or span and reduction for all ten conditions. Keep weak controls and contrary results in the figure. Submit the chosen measured sweep ID as annotations.headline_figure_id. Every other published replacement figure also uses ten conditions. Do not combine conditions measured at incompatible locations just to fill rows.

For Where it sits, author eleven short diagnostic inputs: one readable showcase sentence, followed by ten matched sentences or stems that vary the relevant cue. Use the current neuron's evidence to design these texts. The complete input must show both the cue and the proposed response location. Aim for 7–29 English words and roughly 10–40 model tokens per matched input; the showcase can use up to 44 words when necessary. Retain the exact supplied texts and any meaningful trailing whitespace. These are independently written tests, not excerpts silently cut out of captured corpus inputs. Use paper.family.cases with {text, position:-1}; all positions are measured for the joint heatmap. Natural corpus experiments continue to use their complete captured prefixes.

Use the configured ten upstream and ten downstream neighbours plus the target for the joint traces, giving 21 rows when both sides are available. The host measures these units together on each of the eleven inputs. Give each displayed unit a short label of about 3–7 words supported by its measured response. Submit the complete family once; later rounds can omit family to preserve it. Change it only when a new joint test is needed.

For Circuit, measure the configured thirty downstream reader candidates and retain the measured upstream candidates. Do not replace that reader pool with a shorter list for convenience. Late-layer targets can have fewer available downstream layers. Write a summary of approximately 30 English words, explaining the measured relationship and its scope; individual units and numbers belong in the table.

Read /control/settings.json for the experiment and sampling parameters. The host selects the natural atlas, preserves original token IDs from document start through the anchor, and adds representative cases from each semantic group. Plan the configured rounds: initial tests, then strong counterexamples and matched alternatives.

Every candidate, control and replacement declares its scored token or span and reduction before measurement. Distinguish the changed cue from the response location. Use the same rule for matched conditions. Existing natural cases retain their saved anchor. There is no implicit maximum over a replacement or a full sentence.

State the changed cue, the response token/span and why that location tests the hypothesis in each prediction. A marker's token_offset is an index inside that marker: -1 selects its last token, not the token before the marker. Choose the response location from the hypothesis rather than copying an example offset. To test a response before a value is supplied, score the actual boundary before that value; the last token of an already completed value tests a different hypothesis. A trailing space, when measured, is part of the input. Preserve it explicitly. Cue-token responses and later response-slot responses are separate measurements.

Read the returned selected token, scored_positions and native trace when interpreting a contrast. A near-zero result at one token establishes only a near-zero response at that token; it does not establish that the neuron stayed inactive elsewhere. Keep failed and flat tests visible, while giving the main illustrated figures to tests that meaningfully distinguish the proposed explanations. A strong counterexample belongs among those main figures. A large bar produced only by scaling tiny residual values is not a strong effect.

The host automatically registers every new completed condition, measures target zeroing with same-layer random controls, and computes background local RelP. Same input/anchor/settings reuse their recorded measurements. These follow-ups feed What it does and the measured route data. Natural examples define the atlas; probe assignments remain identified as probes. Choose representative multi-hop display graphs separately.

Blind maximum and Blind minimum are a fixed final measurement step of this stage. The host runs the isolated writer at completion unless matching complete results already exist. You may call scripts/run_tests.py earlier to inspect them; completion reuses that result. The writer sees only the frozen initial explanation, writes the configured positive and negative candidates, and the host measures their full-input maximum and minimum before comparing them with the saved corpus extremes. Route prediction is disabled in this pipeline; leave that report slot as —.

Activation spectrum and Inside the neuron reuse the incoming initial explanation’s predictions on the shared corpus sample. Coordinates and activation bins retain measured values. Read scripts/read_evidence.py --section response_predictions to inspect disagreements and uncertain cases; use this feedback when designing experiments.

Complete with a result summary and annotations using the full JSON example in the API reference. Each annotations.routes value is an object with name and description; each annotations.units value is a label string; section prose uses an object with summary. Use measured route and unit IDs from the results. Completion returns the experimental report with the incoming claim unchanged. Submit after the measurements, spectrum labels and independent tests; this task does not revise the claim.

## Semantic Clusters and raw routes

A route is a numerical contribution pattern, not a semantic category. Keep its measured membership intact in Raw factorization. After the configured measurement rounds and scripts/run_tests.py have returned, read scripts/read_evidence.py --section routes. This gives the complete current natural-training member index. Use --section routes --ids GRAPH_ID ... to read complete observed prefixes, native traces, selected anchors and upstream vectors. The index previews are navigation aids, not sufficient evidence for deciding a member's meaning. Read every member's complete context before assigning it. Read the latest index again if you add experiments after it.

Organize those measured members into reader-facing semantic clusters. Several routes may contribute to one cluster; one route may contain members belonging to different clusters. Use the text, scored position, activation and upstream contributions together. An existing spectrum label is a hint, not a compulsory assignment. Similar text does not prove the same mechanism; similar contributions do not prove the same topic.

Give every member one display cluster, without dropping weak activations, mismatches or counterexamples. Use mixed or unresolved clusters when no supported common account exists. Explain when a numerical route crosses semantic categories. If splitting out a member limits a proposed explanation, retain that limitation explicitly in the affected story; do not use regrouping to make contrary evidence disappear.

For each cluster, state its membership criterion, the shared observations or contribution patterns, and the important exceptions or uncertainty. A name inferred from a few strong examples must not silently describe all remaining members. The tools derive response summaries from the measured members. Explain positive and negative conditions from those measurements; leave unresolved causes explicit.

Choose up to three example_graph_ids from the cluster's actual graph_ids. Show a typical case and, whenever one is informative, a boundary case or counterexample. Do not merely choose the first or highest-activation members. The complete member list remains in the structured report.

Return annotations.clusters as an array of {id, name, story, graph_ids, example_graph_ids}. Use graph_id values from the routes evidence, not saved case IDs or route IDs. The fixed tools derive member_routes, graph counts, attribution-mass shares, upstream summaries and original example values. Do not supply those numbers. Raw routes retain their own annotations.routes name/description mapping.

## Writing

Write a short title in the form “[concept] neuron”, with an optional phrase describing its response. Begin the explanation directly with what this neuron responds to or changes. Refer to the measured unit as “this neuron”; keep its layer and index in the identification fields.

Use ordinary descriptions of the positive and negative responses. Describe sparsity in the activation-distribution discussion, rather than calling the neuron a “sparse feature”. Keep qualifications in sentences instead of stacking them in a long title.

Describe the observations and conclusions directly for the reader.
