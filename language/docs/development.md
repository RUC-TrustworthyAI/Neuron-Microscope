# Development and software checks

Use Linux and Python 3.10 or newer for the full workflow. The offline demo also runs on macOS with Python 3.10+.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r language/requirements/test.txt
python -m language --help
PYTHONPATH=language/tools python -m unittest discover -s language/tests -v
python -m unittest discover -s language/analysis -p 'test_*.py' -v
python language/examples/demo.py --output .local/demo
python -m http.server 8080 --directory .local/demo
```

Open `http://localhost:8080/`. The offline demo uses synthetic measurements to exercise report generation and interaction. RelP tests instantiate a tiny random model on CPU.

The software tests cover preserved input prefixes, data selection, native attribution operations, frozen explanations, saved-output reuse, API accounting, report assembly and layout.

The comparison viewer has a separate test for default opening, compressed-report loading, and scrolling from either pane. Run it with Node.js 18 or newer:

```bash
node --test language/tests/test_comparison_viewer.mjs
```

Workflow tests cover evidence visibility, explanation refinement, frozen rules, shared plot labels, response reuse and time accounting. To exercise report assembly with an existing activation-only package containing static logit effects and peak censuses, set its local path:

```bash
WORKFLOW_TEST_REPORT=/path/to/report PYTHONPATH=language/tools \
  python -m unittest discover -s language/tests -p 'test_workflow*.py' -v
```

These checks render saved evidence and synthetic test conditions on CPU.
