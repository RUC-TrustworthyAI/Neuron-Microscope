# Development and software checks

Use Linux with Bash 4+ and Python 3.10 or newer for the full orchestrator. The offline demo also runs on macOS with Python 3.10+.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/test.txt
bash -n run.sh
PYTHONPATH=tools python -m unittest discover -s tests -v
python -m unittest discover -s analysis -p 'test_*.py' -v
python examples/demo.py --output .local/demo
python -m http.server 8080 --directory .local/demo
```

Open `http://localhost:8080/`. The demo uses synthetic measurements to exercise report generation and interaction; it is not a neuron interpretation or a scientific result. No downloaded model, GPU, API credential or paid request is needed for these checks. RelP tests instantiate a tiny random model on CPU.

The software tests cover preserved input prefixes, data selection, native attribution operations, stage isolation and handoff, saved-output reuse, API accounting, report assembly and layout. Passing them does not establish the scientific validity of an explanation.

Build the agent image separately:

```bash
docker build -t neuron-investigator:local .
docker run --rm --network none neuron-investigator:local \
  python -c 'import deepseek_harness; import shutil; assert shutil.which("jq"); assert shutil.which("rg")'
```

The image installs the public harness SDK and runtime packages. GPU measurement dependencies run on the host, as described in [running.md](running.md); they are not copied into each agent container.

Keep credentials, local settings, model files, activation archives and run records outside Git. A prepared run snapshots only the framework's source directories and selected settings; it does not copy the repository's `.git`, local runs or weights. Scientific evidence and outputs are never edited to make a test pass. Changes to calculation or selection rules should be stated explicitly and exercised with a small fixture before starting a research run.
