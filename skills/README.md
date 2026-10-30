# Agent skills

[Project overview](../README.md) · [Workflow](../docs/method.md) · [Report guide](../docs/reports.md)

These three skills provide task instructions and script clients for the host tools. The launcher mounts only the selected skill directory into each container.

| Skill | Input | Job | Output |
|---|---|---|---|
| [Interpret](interpret/SKILL.md) | Target activation-only HTML and structured evidence, with measured extrema and quantile examples | Read the observed responses and form an initial explanation | Claim and initial HTML handoff |
| [Experiment](experiment/SKILL.md) | Initial explanation, HTML and saved evidence | Plan experiments, invoke measurement tools, organize and annotate results | Measured artifacts, plots and HTML handoff |
| [Revise](revise/SKILL.md) | Explanation and measured experimental handoff | Revise the claim, Criticism and captions from the evidence | Final prose and HTML, with an independent reading |

Each directory follows the same structure:

```text
skill/
  SKILL.md             Task scope and workflow
  references/          API contracts and operating instructions
  scripts/             Python clients for the host tool service
```

The task names `/skills/<name>/SKILL.md`; the agent reads it and uses the linked scripts to request evidence and execute the stage. Instructions are loaded from these files, and evidence is retrieved on demand.

Container lifecycle and handoffs belong to [run.sh](../run.sh). Measurement, evidence retrieval and rendering belong to [tools/](../tools/). The scripts in these skill directories are the agent's interfaces to those fixed implementations.

| Stage | Tool reference | Workflow reference |
|---|---|---|
| Interpret | [API](interpret/references/api-reference.md) | [Workflow](interpret/references/workflows.md) |
| Experiment | [API](experiment/references/api-reference.md) | [Workflow](experiment/references/workflows.md) |
| Revise | [API](revise/references/api-reference.md) | [Workflow](revise/references/workflows.md) |
