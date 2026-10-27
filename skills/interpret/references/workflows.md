# Evidence workflows

Run these commands from this skill directory.

```bash
python scripts/read_evidence.py
python scripts/read_evidence.py --section distribution
python scripts/read_evidence.py --section token_selectivity
python scripts/read_evidence.py --section peak_census
python scripts/read_evidence.py --section gate_value
python scripts/read_evidence.py --section output_effects
python scripts/read_evidence.py --section contexts --offset 0 --limit 4
```

Read further contexts at `next_offset`, filter by a group returned by the index, or request specific IDs:

```bash
python scripts/read_evidence.py --section contexts --group top_tokens --limit 4
python scripts/read_evidence.py --section contexts --ids CASE_ID
```

Base the explanation on the supplied activation evidence. The evidence tool returns saved measurements.

## Submit this stage

```bash
python scripts/submit_interpretation.py <<'JSON'
{"claim":{"title":"","lede":"","one_paragraph":"","negative_pole":"","what_would_change_my_mind":"","confidence":""},"criticism":[],"response_rules":[{"id":"rule_1","name":"Short contextual condition","description":"State the observed textual condition and anchor position that imply this response.","predicted_pole":"positive"},{"id":"no-visible-cue","name":"No visible cue","description":"The context does not clearly support a positive or negative response condition.","predicted_pole":"uncertain"}]}
JSON
```

Use at most three positive and three negative response groups, plus exactly one No visible cue group. Agent 1 submits the initial explanation and bucket definitions; the host immediately performs per-sample labeling. The revision agent reads those saved predictions alongside experimental evidence before revising the explanation and buckets.

Fill the claim from the saved evidence. Detailed experimental Criticism is written in stage three. The script saves the initial explanation and an HTML copy; stop when it returns.
