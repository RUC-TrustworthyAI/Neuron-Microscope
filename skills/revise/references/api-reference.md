# Report API

```bash
python scripts/write_report.py <<'JSON'
{"claim":{},"response_rules":[{"id":"rule_1","name":"Contextual condition","description":"Textual condition and anchor position predicting the response.","predicted_pole":"positive"},{"id":"no-visible-cue","name":"No visible cue","description":"The context does not clearly support a positive or negative response condition.","predicted_pole":"uncertain"}],"criticism":[],"annotations":{"figures":{},"graphs":{},"routes":{},"units":{},"sections":{}}}
JSON
```

Use at most three positive and three negative response groups, plus exactly one No visible cue group. Agent 1 submits the initial explanation and bucket definitions; the host immediately performs per-sample labeling. The revision agent reads those saved predictions alongside experimental evidence before revising the explanation and buckets.

Claim fields: title, lede, one_paragraph, negative_pole, what_would_change_my_mind, confidence.

Approximate English word targets: lede 60–85; one_paragraph 150–180; negative_pole 80–120; what_would_change_my_mind 40–60. The last three prose fields are the three paragraphs of The claim in detail. Do not pad a field to reach a target.

Each Criticism entry has round, attack, prediction, result, verdict, evidence. Put aggregates in result. Each evidence entry pairs text with an individual measured activation.

`verdict` is exactly one of `survived`, `narrowed`, or `rejected`, matching the three report badges. `survived` means the tested claim survived as written; `narrowed` means the finding requires a qualification or revision; `rejected` means the tested claim failed. Put the explanation of that decision in `result`, never in `verdict`. `attack` is the critic's challenge shown beside the badge; `prediction`, `result`, and individual evidence appear when the row is opened.

```json
{
  "criticism": [
    {
      "round": 1,
      "attack": "State the alternative explanation being tested.",
      "prediction": "State the contrasting predictions before reporting the measurements.",
      "result": "Explain the measured outcome and why it narrows the claim.",
      "verdict": "narrowed",
      "evidence": [{"text": "An actual measured example", "activation": 1.0}]
    }
  ]
}
```

The example is structural; use this run's actual text and measurements.

Annotations:

- headline_figure_id: the measured ten-condition common-frame sweep selected for the headline. Omit to preserve the experiment stage's choice.
- graph_ids: exactly three measured display-graph IDs in publication order. These come from multi-hop graphs, independently of the route-atlas IDs used in clusters.
- figures, keyed by experiment ID: title, rationale, caption.
- graphs, keyed by experiment ID: title, why, look_at (list).
- routes, keyed by measured route ID: name, description. These name the raw numerical patterns; they do not define semantic membership.
- clusters (optional): the complete array shown below. Omit to preserve the existing semantic groups and representative examples. Editing routes alone never replaces clusters.
- units, keyed by Lx_Ny: label string.
- sections, keyed by spectrum/search/clusters/anatomy/ablation/family/circuit/tests: summary. Spectrum is approximately 35–55 English words; search 45–70; Circuit approximately 30 words; other introductions 25–50. Anatomy may also include strip_caption. Captions and cluster stories are approximately 35–65 words. These are prose guidelines, not schema size limits.
- anatomy (optional): sentence, pairs and groups using the same saved-case selectors as the experiment skill. This changes the selected existing traces without a model call.

The renderer takes numerical values from saved measurements. Your annotations supply the interpretation. Blind maximum/minimum tests measure the incoming initial explanation. Inside the neuron and Activation spectrum share the initial explanation’s saved response predictions. Use that feedback for revision; their labels retain the original explanation version. Final submission runs the independent review and preserves these predictions, then returns the rendered report path. Finish wording edits in the local JSON before calling write_report.py; wait for the final result and stop.

## Update captions and presentation

After a report has been submitted, render_report.py accepts display annotations only:

```bash
python scripts/render_report.py <<'JSON'
{"annotations":{"figures":{"MEASURED_FIGURE_ID":{"caption":"Updated description of the saved result."}},"sections":{"circuit":{"summary":"Updated introduction based on the saved measurements."}}}}
JSON
```

Supported keys within annotations are figures (title, rationale, caption), graphs (title, why, look_at), sections (summary, strip_caption), headline_figure_id and graph_ids. Use measured IDs. Omitted fields are preserved. This tool makes a report copy and updates the stage's final report path; it preserves the claim, response_rules, Criticism, independent review, measurements and predictions. It makes no API or model calls. A revised explanation or bucket definition belongs in the final write_report.py submission.

## Read evidence

```bash
python scripts/read_evidence.py --section brief
python scripts/read_evidence.py --sections distribution peak_census output_effects
python scripts/read_evidence.py --section index
python scripts/read_evidence.py --section response_predictions --limit 4
python scripts/read_evidence.py --section contexts --group bottom_tokens --limit 4
python scripts/read_evidence.py --section measurements
python scripts/read_evidence.py --section measurements --ids EXPERIMENT_ID
```

The brief returns interpretation, sections, experiment_index, figures, graphs, cluster_definitions and response_predictions. The prediction cases include complete observed prefixes and anchors, with strong directional errors first; use --limit to choose how many cases to read. Each experiment entry includes status, statistics, directory path and JSON file names. Read complete evidence sections together with --sections, or complete contexts and route members together with --ids. These exports preserve the requested evidence. Replace EXPERIMENT_ID with an ID from the experiment list. Full measurements remain in the referenced files. The report HTML is generated by the host.


## Semantic cluster annotation

Read the incoming definitions with --section routes. Use --section routes --ids GRAPH_ID to read a complete measured member. The index includes all natural training route members, including low activations; its context_preview is only for navigation. graph_ids are measured graph identifiers, not case IDs. Counts, shares, member_routes, key units and example values are host-derived.

```json
{
  "annotations": {
    "clusters": [
      {
        "id": "c1",
        "name": "Supported semantic group",
        "story": "What the actual members share, which contribution patterns recur, and what remains contrary or unresolved.",
        "graph_ids": ["MEASURED_GRAPH_ID_A", "MEASURED_GRAPH_ID_B"],
        "example_graph_ids": ["MEASURED_GRAPH_ID_A", "MEASURED_GRAPH_ID_B"]
      }
    ]
  }
}
```

Replace all placeholders with measured IDs and evidence-supported prose. Include the other clusters as well when replacing the array. Keep every member represented; mixed or unresolved groups are permitted. Choose at most three examples per group, retaining informative exceptions. Do not add numeric values.

The response_predictions section returns the frozen explanation version, response_rules, separate feedback/seen/experiment summaries and paginated predictions joined to measured signs and activations. Cases are ordered with strong directional errors first. Use --ids, --group, --offset and --limit as needed. Revise response_rules together with the claim, even when retaining the same rule IDs.
