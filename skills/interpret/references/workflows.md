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
{"claim":{"title":"","lede":"","one_paragraph":"","negative_pole":"","what_would_change_my_mind":"","confidence":""},"criticism":[]}
JSON
```

Fill the claim from the saved evidence. Detailed experimental Criticism is written in stage three. The script saves the initial explanation and an HTML copy; stop when it returns.
