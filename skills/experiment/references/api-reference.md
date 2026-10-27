# Experiments API

Read `/control/settings.json` for the experiment and sampling parameters. Submit hypotheses, texts, source cases and explicit scoring locations.

## Read evidence

```bash
mkdir -p /workspace/reads
python scripts/read_evidence.py --section brief > /workspace/reads/brief.json
python scripts/read_evidence.py --sections distribution peak_census output_effects > /workspace/reads/statistics.json
python scripts/read_evidence.py --section contexts --group top_tokens --offset 0 --limit 4 > /workspace/reads/top_tokens.json
```

The brief combines interpretation, section summaries, experiment_index (including paths and JSON file names), figure and graph descriptions, cluster definitions and prediction feedback. Its prediction cases retain complete observed prefixes and anchors, ordered with strong directional errors first. Use --limit for the number of feedback cases. Use --sections to read several complete evidence sections in one call; use --ids ID ... to fetch several complete contexts or route members together. Choose further reads from the returned IDs and paths. Complete requested records remain available.

Use /workspace/reads for your own exports. /workspace/evidence also contains host-generated input.json and manifest.json; manifest.json.cases is a count, whereas a contexts response has a cases list. Read the context files you exported rather than treating every JSON file in the internal evidence directory as a context page.

Other sections: distribution, token_selectivity, peak_census, gate_value, output_effects, measurements and routes. Request contexts by saved IDs when needed. Use --section measurements to list compact experiment overviews and --section measurements --ids EXPERIMENT_ID to inspect one experiment's status, values, directory and JSON file names. The complete artifacts remain in that directory.

saved_analysis is generated only when existing_only is true. An inference run does not create it. In measured results, anatomy_cases is a count, not a list; follow-up conditions is a count, whereas replacement-sweep conditions is a list. Read the named values directly, and use the referenced raw files for traces and detailed tables.

## Read raw routes and organize semantic Clusters

Run the configured independent tests before this read so the index includes the final natural-atlas members. This read performs no experiments.

```bash
printf '%s\n' '{}' | python scripts/run_tests.py
python scripts/read_evidence.py --section routes > /workspace/reads/routes.json
python scripts/read_evidence.py --section routes --group p1
python scripts/read_evidence.py --section routes --ids MEASURED_GRAPH_ID
```

For run_tests.py, supply an empty JSON object through stdin or --input FILE (see the Independent tests section). Replace p1 and MEASURED_GRAPH_ID with IDs returned by this run.

The index contains every natural training route member, its graph_id, raw route_id, selected token/position/product, existing semantic_group when available, upstream contributors and NMF route weights. context_preview is only a navigation preview. --ids returns complete observed text and native trace; no full-prefix measurement changes. --group filters the member index by a raw route ID. semantic_clusters contains the previous submitted definitions in a revision stage.

Semantic grouping uses graph_id, not the corpus case_id. Account for all index members exactly once in the display grouping. Raw routes are retained independently. Merge routes or split their members when the evidence supports it; retain mixed, unresolved and contrary cases. Names and stories must describe the actual scope and exceptions. Choose up to three examples, including an informative counterexample when present.

## Submit a group

```json
{
  "analysis": "The hypothesis, competing explanation and predicted difference.",
  "round": 1,
  "sweeps": [{
    "title": "A controlled replacement test",
    "pole": "positive",
    "prediction": "How changing the cue should change the specified response.",
    "template": "<fixed prefix> {slot}<single-token response marker><fixed suffix>",
    "replacements": [
      "<candidate 1>", "<candidate 2>", "<candidate 3>", "<candidate 4>",
      "<opposite-pole candidate 1>", "<opposite-pole candidate 2>",
      "<opposite-pole candidate 3>", "<opposite-pole candidate 4>",
      "<unrelated control 1>", "<unrelated control 2>"
    ],
    "score": {
      "location": "marker",
      "text": "<single-token response marker>",
      "occurrence": 0,
      "token_offset": 0,
      "reduction": "single"
    }
  }],
  "graphs": [{
    "title": "A representative saved context",
    "prediction": "Which observed response this graph explains.",
    "case_id": "<ID from evidence>",
    "root": {"kind":"neuron_product"}
  }],
  "paper": {
    "search": [{
      "id": "hypothesis-a",
      "name": "<hypothesis>",
      "pole": "positive",
      "rival": "<alternative>",
      "score": {"location":"marker","text":"<single-token response marker>","token_offset":0,"reduction":"single"},
      "candidates": [{"text":"<candidate input>"}],
      "controls": [{"text":"<matched control input>"}]
    }],
    "atlas": {"cases":["<optional additional saved ID>"]},
    "family": {"cases":[
      {"text":"<short authored showcase>","position":-1},
      {"text":"<matched written input 1>","position":-1},
      {"text":"<matched written input 2>","position":-1},
      {"text":"<matched written input 3>","position":-1},
      {"text":"<matched written input 4>","position":-1},
      {"text":"<matched written input 5>","position":-1},
      {"text":"<matched written input 6>","position":-1},
      {"text":"<matched written input 7>","position":-1},
      {"text":"<matched written input 8>","position":-1},
      {"text":"<matched written input 9>","position":-1},
      {"text":"<matched written input 10>","position":-1}
    ]},
    "anatomy": {"sentence":{"case_id":"<saved ID>"}}
  }
}
```

These strings illustrate the API. Replace them with choices from the target evidence. Send the JSON to `python scripts/run_experiments.py`.

Routes per pole, neighbour counts, attribution node and edge counts, controls and prefix policy are applied by the tools from settings.json. Specify hypotheses, texts and scored locations in the request.

## Scoring

Declare the location and reduction before measuring candidates. Candidates and matched controls use the same rule. A candidate may carry its own explicit score when its frame requires a different marker.

For marker scoring, occurrence is the zero-based occurrence of the literal marker text. token_offset indexes the tokens overlapping that marker: 0 is its first token and -1 is its last token. It does not move to the token before the marker. With no token_offset, the declared reduction applies to all tokens overlapping the marker. The single-token markers below are format examples, not a recommendation to score the beginning or end of every phrase. Choose the actual token from the scientific hypothesis and name it in prediction.

For an upcoming-value hypothesis, the response is before the value begins, not at the last digit or word after the value has already been supplied. One option is to end each matched written input at the intended boundary and use {"location":"token_position","position":-1,"reduction":"single"}; then -1 means the last token of that entire input. Preserve any intended trailing whitespace. The returned selected.token and selected.position identify the measured token. Compare a cue-span score separately when needed; do not substitute a whole-input maximum for a declared response-slot measurement.

```json
[
  {
    "location": "marker",
    "text": "<single-token response marker>",
    "occurrence": 0,
    "token_offset": 0,
    "reduction": "single"
  },
  {
    "location": "token_position",
    "position": 12,
    "reduction": "single"
  },
  {
    "location": "replacement_span",
    "reduction": "max"
  },
  {
    "location": "after_replacement",
    "reduction": "min"
  },
  {
    "location": "input",
    "reduction": "max"
  }
]
```

There is no implicit replacement-span or whole-input maximum. If the cue and a later response slot are both of interest, submit two explicitly named measurements. Natural saved cases keep their original anchor coordinate. The records retain scored_positions and the actual selected token.

## Full original prefixes

A saved case resolves to its original token IDs from document position0 through its anchor, inclusive. The host neither retokenizes it nor drops earlier tokens. The next saved token is recorded as a prediction target. Atlas, family, illustrated saved-case graphs and follow-up measurements share this policy.

Written inputs use the configured maximum token length, without silent truncation. A written graph has `text` and an explicit root `position` or `marker`; a saved graph uses `case_id` and the host's original anchor. Output-logit roots additionally name `token_text`.

## Automatic follow-ups

Every group has one model worker. The host registers the configured positive/negative natural extreme references and each completed sweep, search, atlas or selected-graph condition, then automatically measures:

- Native gate, value and product at the full-prefix anchor.
- Target-neuron zeroing and configured same-layer random-neuron controls.
- Next-token distribution changes and local RelP upstream/downstream vectors.

Identical inputs/anchors/settings reuse their saved registry measurements. The response identifies completed and failed conditions.

The registry, raw logits, contribution vectors and source references are stored under the group's followups directory. New probes receive route assignments from factors fitted on natural training cases. Circuit continues to aggregate the natural atlas. Probe results remain separately identifiable.

Named semantic groups also contribute representative saved anchors, covering each observed sign where available. The host adds pending group representatives before stage two completes. Agent-selected illustrated graphs retain the full multi-hop RelP drawing; other conditions receive background contribution vectors.

## Spectrum and labels

The incoming response_rules define the spectrum's response groups: at most three positive, at most three negative, and one No visible cue group for cases with no distinguishable direction. The host has already frozen these definitions and obtained independent text-only assignments. Use the evidence reader's response_predictions section to inspect errors and ambiguous cases.

To read the current group definitions and counts:

```json
{}
```

```bash
python scripts/label_spectrum.py --input /workspace/request.json
```

The tool returns groups, n_requested, explanation_version, reused, llm_requests and new_model_forwards. It reuses the incoming predictions. Do not submit a new topic taxonomy during this stage. At completion the host predicts new measured contexts under the same explanation, reusing unchanged inputs.

Every spectrum point keeps its observed activation and bin. Colour follows the predicted response group: positive, negative or uncertain. Opposite-sign members are retained as counterexamples. The scatter and spectrum share one prediction for an identical observed prefix and anchor. Full API records are saved by the tools.

## Independent tests

Call `scripts/run_tests.py` once. The writer receives the frozen initial explanation. Candidate count and text length come from settings.json. Its declared metric is the maximum or minimum over the written input. These measured candidates automatically receive the same follow-up calculations.

Route prediction is disabled in this pipeline. Its report slot stays —.

## Complete

The complete example below specifies the input shape. Replace its prose and IDs with measured findings and IDs from this run. It is not an interpretation example. Empty optional mappings may remain {}.

```json
{
  "summary": "Summarize the measured findings and remaining alternatives.",
  "annotations": {
    "headline_figure_id": "MEASURED_SWEEP_ID",
    "graph_ids": ["MEASURED_DISPLAY_GRAPH_ID_A", "MEASURED_DISPLAY_GRAPH_ID_B", "MEASURED_DISPLAY_GRAPH_ID_C"],
    "routes": {
      "MEASURED_ROUTE_ID": {
        "name": "Name supported by the measured route examples",
        "description": "Explain the examples and upstream contributions for this measured route."
      }
    },
    "clusters": [
      {
        "id": "c1",
        "name": "Evidence-supported semantic cluster",
        "story": "Membership criterion, observed shared contribution pattern, and exceptions or unresolved evidence.",
        "graph_ids": ["MEASURED_GRAPH_ID_A", "MEASURED_GRAPH_ID_B"],
        "example_graph_ids": ["MEASURED_GRAPH_ID_A", "MEASURED_GRAPH_ID_B"]
      }
    ],
    "units": {
      "L0_N0": "Description supported by the measured unit evidence"
    },
    "sections": {
      "spectrum": {
        "summary": "Explain semantic group counts and activation ranges as counts of the selected samples."
      },
      "search": {
        "summary": "Explain the measured candidates and matched controls at their declared score positions."
      },
      "clusters": {
        "summary": "Explain the measured route groups and their distinctions."
      },
      "anatomy": {
        "summary": "Explain the measured gate/value/product patterns.",
        "strip_caption": "Describe the selected saved trace."
      },
      "ablation": {
        "summary": "Explain direct output weights and measured neuron-zeroing effects."
      },
      "family": {
        "summary": "Explain the joint measured traces of this neuron and its neighbours."
      },
      "circuit": {
        "summary": "Explain the measured upstream and downstream contributions."
      },
      "tests": {
        "summary": "Explain the measured independent blind tests and their outcomes."
      }
    }
  }
}
```

Save the object to /workspace/completion.json and call:

```bash
python scripts/complete_experiments.py --input /workspace/completion.json
```

Fields:

- summary: the experiment findings as a string.
- annotations.headline_figure_id: a measured replacement sweep ID with ten conditions in one frame, covering likely positive, likely negative and unrelated inputs. The renderer uses its complete measured comparison.
- annotations.graph_ids: exactly three measured multi-hop display graph IDs in the desired order. Use IDs returned by graphs, not the local route-atlas graph_ids used in semantic clusters. Other measured graphs remain in saved artifacts.
- paper.family.cases: eleven explicit written inputs, one showcase plus ten matched variations. Each has text and position. The host measures every token for joint traces. Omit family in later rounds to reuse the completed family. Use the configured neighbours; unit labels are provided at completion.
- annotations.routes: keyed by actual measured route ID, such as a route returned in measured_sections.routes. Each value is an object with name and description. A route value is not a plain string.
- annotations.clusters: a complete array of semantic cluster definitions in the shape above. Each graph_id comes from the latest routes index. The measured membership may cross raw route boundaries. Counts, shares, key units and example activations are derived by the host. Submitted members are never removed from raw routes; any unassigned measured members remain visible as unassigned contexts.
- annotations.units: keyed by a measured Lx_Ny identifier. Each value is a label string, not a name/description object. L0_N0 above only illustrates the key syntax; replace it with a measured unit or leave units empty.
- annotations.sections: keyed by the section names shown above. Use an object with summary. Anatomy may also contain strip_caption. Plain strings are supported for section prose, but the complete example uses summary objects consistently.
- Put output effects and zeroing findings in ablation (What it does); upstream/downstream contributions belong to circuit.
- This call connects saved numerical data to the report, adds the configured pending follow-ups and blind tests if needed, labels newly measured points, and renders HTML. A previous successful blind test is reused. Repeated completion calls can regenerate HTML, so submit the finished annotations rather than probing the format by rendering trial pages.
- The host preserves the incoming claim for stage three. This stage's summary and annotations do not replace the final claim revision.


Existing-only runs retain their existing-only workflow: they do not submit numerical experiments or automatic GPU follow-ups.

## Frozen-explanation response predictions

Use `python scripts/read_evidence.py --section response_predictions --limit 4` for prediction-versus-measurement feedback. The response includes the frozen claim and rules, separate feedback/seen/experiment counts, and full-prefix cases ordered with strong directional errors first. Use `--ids`, `--group`, `--offset` and `--limit` to inspect cases. Keep the initial claim and response_rules fixed in this stage; design experiments that clarify their errors and uncertain conditions.
