# Reading a neuron report

[Home](../../README.md) · [Method](method.md) · [Run guide](running.md) · [Data format](data.md)

A report presents the frozen explanation, its supporting measurements, counterexamples and remaining uncertainty. Code assembles the HTML from measured data and structured LLM outputs.

## How to interpret the plots

- **Activation** is the native value before the down projection: `SiLU(raw gate) × up`. Saved native products are used directly rather than reconstructed from rounded displayed values. A negative raw gate is not a hard zero: SiLU can produce a small negative value, so points can have nonzero activation in the shaded left half of the anatomy plot.
- **Scatter colours** show blind predictions from the explanation refined and frozen before experiments: orange for positive, blue for negative, and grey for uncertain. The predictor sees the explanation, rules, complete observed prefix, anchor and nearby continuation, with activations hidden. The coordinates retain measured gate and up. The same samples inform explanation refinement, so their agreement scores provide development feedback rather than a held-out evaluation.
- **Shared sampling** allocates 500 positions equally across nonempty activation bins and selects evenly spaced ranks from each bin’s complete corpus population. Quota shortfalls are redistributed only when a bin’s population is too small. Spectrum and Inside the neuron use the same labeled positions. The full-corpus histogram remains complete; the sample itself is not a density estimate. [Sampling and missing-label handling](data.md#coordinates-and-sampling) describe the procedure.
- **Attribution** uses modified-derivative RelP rules. It is distinct from a native intervention; see [the method's limits](method.md#relp-and-intervention-scope).
- **Costs** are token-price estimates and measured execution records. They are not the provider's account invoice or a measured price for a different GPU setup.

## Where each section comes from

The initial explanation and two revisions each receive blind labels. Among versions with directional accuracy >90%, the highest coverage is selected; otherwise the highest accuracy is selected. That explanation and its corresponding labels are frozen before model experiments. Report analysis interprets the resulting measurements while retaining that frozen account.

| Report section | Evidence and computation | Main implementation |
|---|---|---|
| The claim in detail | Explanation refined using sampled measurements and prediction feedback, then frozen before experiments | [interpretation.py](../workflow/interpretation.py) |
| Figures for this neuron | Controlled contrasts with complete input texts, short condition labels and native response traces | [measure.py](../workflow/measure.py), [report_layout.py](../tools/report_layout.py) |
| Three graphs | RelP graphs at the strongest positive search case, strongest negative search case and strongest absolute control case | [run.py](../workflow/run.py), [relp.py](../tools/relp.py) |
| Activation distribution, exemplars and Peak census | Saved token and document extrema, corpus statistics and sampled quantile bands | [build_reports.py](../analysis/build_reports.py), [saved_display.py](../tools/saved_display.py) |
| Token selectivity | Six rankings based on fractions of each neuron's positive or negative extreme; minimum token frequency 50 | [report_token_fields.py](../analysis/report_token_fields.py), [token_tables.js](../analysis/token_tables.js) |
| Activation spectrum | Measured bins and the final independent predictions from the refined explanation | [sampling.py](../workflow/sampling.py), [labels.py](../workflow/labels.py) |
| Activation search | LLM-proposed texts, successive measured feedback and full-input extrema | [run.py](../workflow/run.py), [measure.py](../workflow/measure.py) |
| Clusters | Semantic memberships over measured natural and written atlas cases; unassigned cases remain in an unresolved group | [revision.py](../workflow/revision.py), [render.py](../workflow/render.py) |
| Inside the neuron | Measured gate/up coordinates and the same labeled positions and frozen predictions as Activation spectrum | [labels.py](../workflow/labels.py) |
| What it does | Saved static logit effects plus selected native interventions and controls | [interventions.py](../workflow/interventions.py), [model_tools.py](../tools/model_tools.py) |
| Where it sits | Selected neighbours and joint responses on the same inputs | [paper_sections.py](../tools/paper_sections.py) |
| Circuit | Local RelP contributions pooled over the selected attribution atlas | [relp.py](../tools/relp.py), [paper_sections.py](../tools/paper_sections.py) |
| Blind maximum / minimum | Independent explanation-only writer produces probes; the target model scores full-input extrema | [run.py](../workflow/run.py), [prompts.py](../workflow/prompts.py) |
| Criticism | Experimental support, contradictions and unresolved patterns with links to measured cases | [revision.py](../workflow/revision.py), [render.py](../workflow/render.py) |
| Interpretation costs | Execution intervals, model operations and recorded API usage | [state.py](../workflow/state.py), [render.py](../workflow/render.py) |
| An agent's read, in full | Independent LLM assessment of the frozen account and its experimental evidence | [revision.py](../workflow/revision.py) |

Appendix remains unfilled. The final layout places Criticism before Interpretation costs, then Appendix, with the full independent reading last.

Publication adds a local original/report comparison through [comparison.py](../tools/comparison.py) when a reference snapshot is available. It opens in side-by-side mode by default, with the published reference on the left and the reproduction on the right. Scrolling either pane synchronizes both reports. Close the comparison, or use `?compare=0`, to read the reproduction alone. The headline, independent reading, Peak census column headings, Top/Bottom examples, quantile samples and other matching sections have independent alignment anchors. The [shared display helper](../tools/exemplar_display.py) limits preceding context for exemplars and cluster examples without changing measured prefixes. Reference pages are supplied separately and used only in the published export.

The [display helper](../tools/report_density.js) initially shows three custom figures, eight cluster cards and eight Criticism rounds, with additional entries expandable. Quadrant tables initially show eight groups. Circuit and Family tables display up to thirty upstream and thirty downstream units selected from attribution results. Top, Bottom and quantile examples initially show five, four and five examples respectively; each cluster displays up to three selected examples. Full saved records remain available in the report artifacts.

Exemplars and cluster examples preview the outlined anchor with up to 32 preceding tokens and expand to the saved context. Activation spectrum examples show up to 128 tokens before the first highlighted activation, or before the anchor for weak responses. These display windows do not change measured prefixes. Search and cluster activation badges use saturated sign colours; token shading represents activation magnitude. Spectrum and anatomy group colours represent the frozen explanation's predicted positive, negative or uncertain response. Cluster colours summarize the measured responses of their members.

The head figure comes from the controlled test with the highest positive response when positive rules exist, otherwise the strongest negative response; response range breaks ties. Each row uses the signed largest absolute native product within its saved scoring span. A contrast retains the same template and scoring rule across conditions. This display quantity is derived from retained traces; the original experiment reductions and results remain unchanged. Claim fields, section summaries and captions follow the output contracts and word targets in `language/workflow/contracts.py` and `language/workflow/prompts.py`.

The Family heatmap displays the selected measured sentence, with rows for the target and attribution-linked upstream and downstream neurons. Steering inputs reuse native traces from completed graph measurements and show an expandable activation context. Attribution-axis labels prioritize positions carrying measured graph nodes; complete observed inputs and node/edge values remain in the saved measurements.

## Inspect an underlying result

The rendered HTML is the presentation layer. Each workflow run also keeps:

```text
config.json                         Input and execution settings
sample/                             Sampled positions, contexts and measurements
evidence.json                       Input to the explanation model
initial-labels/                     Predictions from the initial explanation
refined-labels/, refined-labels-2/  Predictions from each revised explanation
interpretation-history.json         Version metrics and selected round
interpretation.json                 Selected explanation frozen before experiments
predictions.json                    Selected version’s predictions shared by both plots
api/                                Requests, responses, usage, cost and timing
measurements/                       Search, controls and blind-test measurements
atlas/, graphs/                     Attribution, family and intervention data
artifacts.json                      Structured data passed to the renderer
report/                             Final HTML and report data
status.json                         Current phase and output path
```

A report may have unavailable sections when the corresponding measurements have not been made or have failed; descriptive prose does not substitute for that evidence. Use the [run guide](running.md#records) to locate the records and the [method map](method.md) to trace their computation.

Independent text predictions use `language/workflow/prompts.py`. Response rules come from the explanation refined before experiments. Both plots retain measured coordinates and those frozen predictions. Labeling uses five concurrent batches of at most 100. The supplied `llm.max_concurrent_requests: null` imposes no application-level API cap; a numeric limit is enforced through shared API slots on one host.
