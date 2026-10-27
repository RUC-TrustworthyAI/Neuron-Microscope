# Report workflow

Read the evidence brief and the complete cases needed for revision. Finish prose edits in a local JSON file before the final submission. From this skill directory:

```bash
python scripts/write_report.py <<'JSON'
{"claim":{},"criticism":[],"annotations":{"figures":{},"graphs":{}}}
JSON
```

Fill the fields described in the API reference. Submission runs an independent review and response predictions, saves the interpretation and annotations, and renders the measured figures and graphs. Wait for the report path and stop. Use a JSON editor for draft changes before submitting.

For a subsequent display-only update to an already submitted report:

```bash
python scripts/render_report.py --input /workspace/captions.json
```

The file contains an annotations object with figures, graphs, sections, headline_figure_id or graph_ids, as needed. This operation updates captions, introductions and displayed selections using saved measurements. It preserves the claim, response rules, independent review and per-sample predictions and makes no API or model calls.

## Long tools

Tool scripts wait for the operation to finish, then return its saved result. Continue only after that response.
