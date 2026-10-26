# Reading a neuron report

[Home](../README.md) · [Method](method.md) · [Run guide](running.md) · [Data format](data.md)

A report follows one investigation from saved evidence to a revised claim. The HTML is assembled by code from measurements and agent submissions.

## How to interpret the plots

- **Activation** is the native value before the down projection: `SiLU(raw gate) × up`. Saved native products are used directly rather than reconstructed from rounded displayed values. A negative raw gate is not a hard zero: SiLU can produce a small negative value, so points can have nonzero activation in the shaded left half of the anatomy plot.
- **Semantic colours** use the group's explicit `positive`, `negative` or `either` annotation. Colour is not inferred from the median activation, and it is not interchangeable with the point's plotted quadrant.
- **Scan sampling** first draws up to 40 candidates per amplitude interval, then retains up to 256 in proportion to candidate-bin counts. Combined with extrema and experimental points, this shows selected evidence rather than a population density estimate.
- **Attribution** uses modified-derivative RelP rules. It is distinct from a native intervention; see [the method's limits](method.md#relp-and-intervention-scope).
- **Costs** are token-price estimates and measured execution records. They are not the provider's account invoice or a measured price for a different GPU setup.

## Where each section comes from

“Preparation” below means saved-data processing before Agent 1. “Stage 2” means measurements performed by fixed tools in response to the experiment agent. “Stage 3” means revision from that measured handoff.

| Report section | Evidence and computation | Agent contribution / stage | Main implementation |
|---|---|---|---|
| The claim in detail | Target report and experimental results | Initial hypothesis in Stage 1; revised claim in Stage 3 | [interpret skill](../skills/interpret/SKILL.md), [revise skill](../skills/revise/SKILL.md) |
| Figures for this neuron | Measured text replacements and their native activation traces | Stage 2 designs contrasts; Stage 3 revises captions | [model_tools.py](../tools/model_tools.py), [report_layout.py](../tools/report_layout.py) |
| Attribution graphs | RelP graphs rooted at explicit token positions; full scoring inputs retained | Stage 2 chooses cases; Stage 3 interprets the measured graph | [relp.py](../tools/relp.py), [model_tools.py](../tools/model_tools.py) |
| Activation distribution, exemplars and Peak census | Saved native activations, token extrema, document extrema and sampled quantile bands | Fixed saved-data computation; no LLM generates the activation values | [build_reports.py](../analysis/build_reports.py), [saved_display.py](../tools/saved_display.py) |
| Token selectivity | Six rankings based on fractions of each neuron's positive or negative extreme; minimum token frequency 50 | Fixed computation from the saved corpus | [report_token_fields.py](../analysis/report_token_fields.py), [token_tables.js](../analysis/token_tables.js) |
| Activation spectrum | Saved amplitude intervals and context labels | LLM semantic grouping of observed cases; fixed binning and display | [spectrum.py](../tools/spectrum.py) |
| Activation search | Agent-proposed texts scored by the target model | Stage 2 proposes and measures; Stage 3 discusses the results | [paper_sections.py](../tools/paper_sections.py) |
| Clusters | Explicit semantic memberships over measured natural-atlas graphs; raw NMF routes remain separate | Stage 2 groups members and selects representative/contrary examples; Stage 3 reviews them; host derives numerical fields | [semantic_clusters.py](../tools/semantic_clusters.py), [paper_sections.py](../tools/paper_sections.py) |
| Inside the neuron | Raw gate and up/value at selected measured points; captured native product | Scan labels before Stage 1; labels for new experiment points in Stage 2; prose in Stage 3 | [initial_scan.py](../tools/initial_scan.py), [anatomy_selection.py](../tools/anatomy_selection.py), [anatomy_semantics.py](../tools/anatomy_semantics.py) |
| What it does | Available saved output effects plus selected native interventions and controls | Stage 2 measures effects; Stage 3 explains their scope | [model_tools.py](../tools/model_tools.py), [followups.py](../tools/followups.py) |
| Where it sits | Selected neighbours and their joint responses on the same inputs; saved coactivation where available | Stage 2 plan and measurements; Stage 3 description | [paper_sections.py](../tools/paper_sections.py), [saved_analysis.py](../tools/saved_analysis.py) |
| Circuit | Local RelP contributions pooled over the selected attribution atlas | Stage 2 measurements; Stage 3 interpretation | [relp.py](../tools/relp.py), [paper_sections.py](../tools/paper_sections.py) |
| Blind maximum / minimum | An independent writer receives the frozen explanation, writes candidates, and the target model scores them | Fixed Stage 2 completion step; Stage 3 reports the result | [blind_tests.py](../tools/blind_tests.py), [stage_session.py](../tools/stage_session.py) |
| Criticism | Measured examples and experimental outcomes | Agent-written attacks, results and concrete evidence in Stage 3 | [revise skill](../skills/revise/SKILL.md), [report.py](../tools/report.py) |
| Interpretation costs | Stage records, worker telemetry and API usage estimates | Fixed code after revision | [report_costs.py](../tools/report_costs.py) |
| An agent's read, in full | Revised explanation and its measurement summary | Independent LLM reader called during Stage 3 completion | [paper_reading.py](../tools/paper_reading.py) |

Appendix remains unfilled. The final layout places Criticism before Interpretation costs, then Appendix, with the full independent reading last.

Publication optionally adds a local original/report comparison through [comparison.py](../tools/comparison.py). The headline, independent reading, Peak census column headings, Top/Bottom examples, quantile samples and other matching sections have independent alignment anchors. The [shared display helper](../tools/exemplar_display.py) limits preceding context for exemplars and cluster examples without changing measured prefixes. Reference pages are supplied separately and used only in the published export.

## Inspect an underlying result

The rendered HTML is the presentation layer. A prepared run also keeps:

```text
manifest.json                       Frozen settings and target neurons
framework-source/                   Source used by the run
logs/api/                           Complete API requests, responses and usage
units/Lx_Ny/
  01-interpretation/
    preparation/initial-scan/        Saved scan and semantic-labeling records
  02-experiments/workspace/          Tool requests, measured arrays and plots
  03-revision/workspace/             Revised explanation and independent reading
  */handoff/                        JSON, HTML and artifacts passed to the next stage
  PIPELINE_TERMINAL.json             Final state and report path
```

A report may have unavailable sections when the corresponding measurements have not been made or have failed; descriptive prose does not substitute for that evidence. Use the [run guide](running.md#records) to locate the records and the [method map](method.md) to trace their computation.
