"""One configuration for new launches, tools and stage handoffs."""
import copy
import json
import os
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'settings.json'

def load():
    path = Path(os.environ.get('NEURON_MICROSCOPE_SETTINGS', PATH)).resolve()
    config = json.loads(path.read_text())
    return config

def paths(config):
    """Resolve host inputs once, before the source/configuration snapshot is made."""
    base = Path(os.environ.get('NEURON_MICROSCOPE_ROOT', PATH.parent)).resolve()
    if os.environ.get('NEURON_MICROSCOPE_SETTINGS'):
        base = Path(os.environ['NEURON_MICROSCOPE_SETTINGS']).resolve().parent
    defaults = {'reports': 'data/reports', 'activations': 'data/activations',
                'model': 'models/subject', 'state': '.local', 'references': 'data/reference-pages'}
    resolved = {k: str((base / v).resolve()) for k, v in
                {**defaults, **config.get('paths', {})}.items()}
    resolved.setdefault('tokenizer', str(Path(resolved['model']) / 'tokenizer.json'))
    return resolved

def agent_settings(config):
    """Only scientific settings belong in an agent's context."""
    hidden = {'paths', 'execution', 'llm', 'pricing', 'gpu_groups',
              'queue_seconds', 'experiment_seconds', 'runtime'}
    return {k: v for k, v in config.items() if k not in hidden}

def effective(config):
    # Prepared batches carry their own frozen copy.
    settings=config['settings'] if 'settings' in config else load()
    return copy.deepcopy({k:v for k,v in settings.items()
                          if k not in {'paths','execution','llm','pricing'}})

def request_settings(request):
    return effective(request)
