# Experiments API

Read `/control/settings.json` for the experiment and sampling parameters. Submit hypotheses, texts, source cases and explicit scoring locations.

## Read evidence

```bash
mkdir -p /workspace/reads
python scripts/read_evidence.py --section contexts --group top_tokens --offset 0 --limit 4 > /workspace/reads/top_tokens.json
```

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
    "template": "<fixed prefix> {slot} <fixed response marker>",
    "replacements": ["<candidate>", "<matched alternative>"],
    "score": {
      "location": "marker",
      "text": "<fixed response marker>",
      "occurrence": 0,
      "token_offset": -1,
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
      "score": {"location":"marker","text":"<response marker>","token_offset":-1,"reduction":"single"},
      "candidates": [{"text":"<candidate input>"}],
      "controls": [{"text":"<matched control input>"}]
    }],
    "atlas": {"cases":["<optional additional saved ID>"]},
    "family": {"cases":["<saved ID>"]},
    "anatomy": {"sentence":{"case_id":"<saved ID>"}}
  }
}
```

These strings illustrate the API. Replace them with choices from the target evidence. Send the JSON to `python scripts/run_experiments.py`.

Routes per pole, neighbour counts, attribution node and edge counts, controls and prefix policy are applied by the tools from settings.json. Specify hypotheses, texts and scored locations in the request.

## Scoring

Declare the location and reduction before measuring candidates. Candidates and matched controls use the same rule. A candidate may carry its own explicit score when its frame requires a different marker.

```json
[
  {
    "location": "marker",
    "text": "<fixed response marker>",
    "occurrence": 0,
    "token_offset": -1,
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

Define the complete group catalog from this neuron's evidence before submitting it. The following is a complete API-shape example, not proposed semantic groups or interpretation examples. Replace it with the groups supported by the target evidence; add, remove or split groups as needed. Include an observed opposite pole only when the evidence supports one.

```json
{
  "groups": [
    {
      "id": "positive-context",
      "name": "Supported positive context",
      "description": "Describe the recurring positive-response context established from this neuron's saved evidence.",
      "pole": "positive"
    },
    {
      "id": "negative-context",
      "name": "Supported negative context",
      "description": "Describe the recurring negative-response context established from this neuron's saved evidence.",
      "pole": "negative"
    },
    {
      "id": "background",
      "name": "Background or unclear context",
      "description": "Saved contexts that do not fit the supported response groups, including ambiguous or background material.",
      "pole": "either"
    }
  ]
}
```

Save that object to /workspace/groups.json and call:

```bash
python scripts/label_spectrum.py --input /workspace/groups.json
```

### Fields and scope

- Fresh request: groups is the group-definition array. Submit a nonempty, complete set of definitions.
- Each definition: id is a unique stable string; name is its displayed name; description states the semantic membership criterion; pole is positive, negative or either. Supply all four fields. id and name are read by the tool; the definitions and pole drive classification and colours.
- The sample set is all unique positions in the target report's points: extrema, quantile examples and scatter positions. It is neither only the cases used to propose a group nor all tokens in the activation archive.
- There is no sample-count or pilot-size argument. In particular, groups[*].case_ids does not select the labeling sample set. One group still sends all saved positions for classification.
- Requests are internally batched in groups of 24 cases. A fresh call makes ceil(n / 24) labeler requests for n saved positions, regardless of the number of semantic groups.
- Numerical signs alone do not identify semantic groups. Independent blind tests remain separate from this scatter.
- A result reports status, n_requested, n_assigned, groups, missing_case_ids, batch_count, n_retained, n_new_requested, resume_from and new_model_forwards. Unassigned cases do not imply a smaller input sample.

### Reuse and continuation

- Repeating the same group definitions in the same stage workspace returns an existing recorded result when the groups match, including a partial result. This does not fill its missing labels.
- Matching includes group order and the submitted definition fields. A change to IDs, wording, poles, group membership criteria or the group list is not an unchanged request.
- To continue missing labels under unchanged definitions, use the exact resume_from directory returned by the previous result:

```json
{"resume_from": "spectrum-003"}
```

Replace spectrum-003 with the returned directory. The tool loads that directory's saved groups and assignments, retains existing assignments, and labels only still-unassigned cases. resume_from takes the saved definitions; supplying changed groups alongside it does not replace those definitions.
- To revise the group definitions, submit a fresh complete groups array. The fresh run labels all saved positions; do not expect labels from a different taxonomy to be reused.
- Cache lookup is within the current stage workspace, not an automatic import from another run.

## Independent tests

Call `scripts/run_tests.py` once. The writer receives the frozen initial explanation. Candidate count and text length come from settings.json. Its declared metric is the maximum or minimum over the written input. These measured candidates automatically receive the same follow-up calculations.

Route prediction is disabled in this pipeline. Its report slot stays —.

## Complete

The complete example below specifies the input shape. Replace its prose and IDs with measured findings and IDs from this run. It is not an interpretation example. Empty optional mappings may remain {}.

```json
{
  "summary": "Summarize the measured findings and remaining alternatives.",
  "annotations": {
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
        "pole": "positive",
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
- annotations.routes: keyed by actual measured route ID, such as a route returned in measured_sections.routes. Each value is an object with name and description. A route value is not a plain string.
- annotations.clusters: a complete array of semantic cluster definitions in the shape above. Each graph_id comes from the latest routes index. The measured membership may cross raw route boundaries. Counts, shares, key units and example activations are derived by the host. Submitted members are never removed from raw routes; any unassigned measured members remain visible as unassigned contexts.
- annotations.units: keyed by a measured Lx_Ny identifier. Each value is a label string, not a name/description object. L0_N0 above only illustrates the key syntax; replace it with a measured unit or leave units empty.
- annotations.sections: keyed by the section names shown above. Use an object with summary. Anatomy may also contain strip_caption. Plain strings are supported for section prose, but the complete example uses summary objects consistently.
- Put output effects and zeroing findings in ablation (What it does); upstream/downstream contributions belong to circuit.
- This call connects saved numerical data to the report, adds the configured pending follow-ups and blind tests if needed, labels newly measured points, and renders HTML. A previous successful blind test is reused. Repeated completion calls can regenerate HTML, so submit the finished annotations rather than probing the format by rendering trial pages.
- The host preserves the incoming claim for stage three. This stage's summary and annotations do not replace the final claim revision.


Existing-only runs retain their existing-only workflow: they do not submit numerical experiments or automatic GPU follow-ups.
