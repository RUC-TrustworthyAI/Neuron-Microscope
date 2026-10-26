# Report workflow

Read the previous stage at `/inputs/handoff/handoff.json` and `report/index.html`, then from this skill directory:

```bash
python scripts/write_report.py <<'JSON'
{"claim":{},"criticism":[],"annotations":{"figures":{},"graphs":{}}}
JSON
```

Fill the fields described in the API reference. Submission runs an independent review of the report, saves your interpretation and annotations, and renders the measured figures and graphs. Wait for the tool to return its report path.

## Long tools

Tool scripts wait for the operation to finish, then return its saved result. Continue only after that response.
