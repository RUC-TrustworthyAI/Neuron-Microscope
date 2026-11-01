"""Create a local investigation configuration from the language settings."""
import argparse
import json
from pathlib import Path


def configuration(neuron, model, report, activations, gpus):
    settings = json.loads((Path(__file__).resolve().parents[1] / 'settings.json').read_text())
    settings['gpu_groups'] = [gpus]
    return {
        'neuron': neuron,
        'model': str(Path(model).resolve()),
        'report': str(Path(report).resolve()),
        'activations': str(Path(activations).resolve()),
        'gpus': gpus,
        'settings': settings,
        'llm': dict(settings['llm'], pricing=settings['pricing']),
        'api_slots': str(Path('.local/api-slots').resolve()),
        'gpu_lock': str(Path('.local/gpu-locks').resolve() / ('gpu-' + '-'.join(map(str, gpus)) + '.lock')),
        'max_rules_per_pole': 5,
        'max_refinements': 2,
        'search_rounds': 3,
        'search_regions': 6,
        'candidates_per_region': 16,
        'atlas_natural': 20,
        'atlas_written': 20,
        'batch_size': 16,
        'batch_tokens': 8192,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neuron', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--activations', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = configuration(args.neuron, args.model, args.report, args.activations, args.gpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(config, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
