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



def effective(config):
    # Prepared batches carry their own frozen copy.
    settings=config['settings'] if 'settings' in config else load()
    return copy.deepcopy({k:v for k,v in settings.items()
                          if k not in {'paths','execution','llm','pricing'}})

def request_settings(request):
    return effective(request)
