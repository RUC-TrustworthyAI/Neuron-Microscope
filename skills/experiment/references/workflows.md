# Experiment workflow

Read the incoming explanation, saved evidence and /control/settings.json.

1. Identify competing accounts for the observed positive and negative responses. Read complete observed prefixes. Following text is a prediction target.
2. Read the incoming response predictions and their frozen rules with scripts/read_evidence.py --section response_predictions. Activation spectrum and Inside the neuron share these saved labels. Use their disagreements with measurements to design tests.
3. Use the configured search rounds. In the first round, submit controlled hypotheses with explicit scored locations. Read the resulting measurements. In the next round, choose strong counterexamples and matched alternatives that distinguish the surviving accounts.
4. Reuse prior completed conditions. The host registers new conditions and runs their full-prefix ablations and background RelP automatically.
5. Choose positive and negative illustrated graphs from the evidence; include an output-logit root where useful. These receive the configured multi-hop graph calculation.
6. Run scripts/run_tests.py before organizing Clusters; matching completed tests are reused. This also finishes the pending semantic-group representatives through the existing measurement tools. Read the latest natural-atlas routes with scripts/read_evidence.py --section routes, then read all member contexts by graph ID. Keep raw routes separate from semantic clusters; merge or split their members only as supported by text and contribution evidence. Preserve contrary and unresolved members and describe the limitations they expose. Select representative and informative exception examples. If more experiments are added, read the latest membership again before submission.
7. Complete the experiment handoff with measured summaries and captions, using the complete_experiments JSON example: route values are name/description objects, clusters is an array with explicit graph_ids and example_graph_ids, unit values are strings, and section prose uses summary objects. The host generates the HTML. Stage three performs the final revision.

Each plan states the changed cue, fixed context, scored token/span, predicted direction and alternative explanation. Keep the same scoring rule for candidates and matched controls. A word response and a later prediction slot are distinct measurements.

Use the sampling and measurement parameters supplied in settings.json. Base hypotheses on the supplied target evidence.

## Existing-only runs

When existing_only is true, read saved coactivation and prior measurements, label semantic groups and prepare unmeasured text proposals through scripts/prepare_tests.py. No model inference or automatic GPU follow-up is available. Unmeasured results remain —. Semantic associations do not establish causal edges.

## Long tools

Tool scripts wait for the host operation to finish, then return its saved result. GPU allocation and waiting are handled by the host. A failed result stays in the records; do not repeat it merely to obtain a passing result.

## Command-line input

Every script supports --help. Read-evidence accepts --section, --group, --ids, --offset and --limit, or a JSON object through stdin. Other scripts accept JSON through redirected stdin or --input FILE. Unsupported flags and invalid JSON exit immediately with a usage error. Tools wait for the completed operation; no progress-polling loop is needed.
