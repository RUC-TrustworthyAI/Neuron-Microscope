# Method and implementation

The subject is a native dense-Qwen3 MLP neuron. The research agent uses a separate DeepSeek model accessed through its API. A GPU group holds one sharded copy of the subject model, not several different models.

| Operation | Code | Origin of content |
|---|---|---|
| Source/configuration snapshot and containers | `run.sh`, `tools/launch_files.py` | Fixed host code |
| Scan before explanation | `tools/initial_scan.py`, `anatomy_scan.py`, `anatomy_selection.py` | Saved activations; fixed sampling |
| New scan/probe semantics | `tools/anatomy_semantics.py` | DeepSeek annotations of observed prefixes |
| Initial explanation | `skills/interpret/` | Agent 1 |
| On-demand context/statistics | `tools/evidence.py`, `stage_session.py` | Saved target measurements |
| Replacement sweeps | `tools/model_tools.py` | Agent designs text; model supplies numbers |
| Full-prefix scoring and automatic followups | `tools/experiment_defaults.py`, `followups.py` | Fixed selection and saved conditions |
| Activation search / family / atlas | `tools/paper_sections.py` | Agent plan and measured native responses |
| Multi-hop attribution and local contributions | `tools/relp.py` | Model derivatives with RelP backward rules |
| Blind maximum/minimum | `tools/blind_tests.py` | Separate explanation-only writer; model scoring |
| Semantic Clusters over raw routes | `tools/semantic_clusters.py` | Agent assigns graph members and examples; host derives numerical summaries |
| Revision and Criticism | `skills/revise/` | Agent 3 from the measured handoff |
| Independent reader | `tools/paper_reading.py` | Separate LLM reading and classification |
| HTML structure, plots and costs | `tools/report*.py`, `analysis/report.html` | Fixed rendering of submitted data and prose |
| API requests, tokens and estimates | `tools/api_meter.py` | Host proxy; full request/response records |

The three skills are files mounted into fresh containers and explicitly named in each short task. The agent loads the instructions and calls their scripts over a Unix socket. Only one skill is present per stage.

Measurements use native BF16 channels, full captured prefixes, explicit score positions and signed response poles. Corpus sample counts are not population semantic coverage. A coherent description, a predictive explanation and a causal account remain different claims.

## RelP and intervention scope

`tools/relp-source.json` pins the Transluce source revision and describes adaptations. Native HF forward values are retained while the backward pass freezes RMS scales and attention Q/K, uses the SiLU identity rule and the gated half rule. Local MLP-to-MLP contributions block intervening MLP outputs. The displayed graph additionally weights a local edge by its receiver's influence on the root.

These are modified-derivative attribution measurements. They are not exact additive causal shares. Separate native zeroing and scaling measurements evaluate the selected neuron at the explicit token position. Family measurements and upstream/downstream tables have their own saved selection and aggregation scope; they do not imply an exhaustive circuit census.

## Semantic Clusters and raw routes

NMF components summarize numerical attribution patterns; a shared component is not proof of a shared semantic topic. The experiment skill reads the complete latest natural-training member index after its independent tests, then reads full member contexts by graph ID. It explicitly assigns members to semantic clusters, which may merge or split raw routes. Names and stories must account for the actual text, scored position, activation and measured contributions, including contrary or weak cases.

The raw numerical factorization remains intact. Fixed code derives cluster counts, shares of absolute retained contribution mass, pooled upstream summaries and original example values. Unassigned measured members remain visible. The revision skill preserves the submitted grouping unless it supplies a replacement definition; merely changing a route name does not change semantic membership. Regrouping must not hide counterexamples or imply that related text proves a shared mechanism.

## Independent inputs

Fresh interpretation receives no official neuron description or historical explanation. Reference-style HTML is a presentation format. The independent blind writer receives the frozen explanation, not the natural examples or corpus extremes. Its measured results stay attached to the explanation version it tested. Revision does not run new target-model experiments.

## Records and limits

Every model/API operation retains inputs and outputs. Failed or incomplete measurements remain recorded. Software tests exercise the implementation; scientific evidence comes from the saved measurements.
