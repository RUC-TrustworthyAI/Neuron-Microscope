"""Command-line modes, follow-up integration and report publication."""
import argparse
import os
import sys
from pathlib import Path

from .state import Timeline, lock, now, read, write


def main():
    from .run import run

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--config', type=Path)
    mode.add_argument('--ablate-from', type=Path, metavar='INVESTIGATION',
                      help='measure and render expanded ablations from a completed investigation')
    mode.add_argument('--anatomy-from', type=Path, metavar='INVESTIGATION',
                      help='render paired Anatomy traces from a completed investigation')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--backend-tools', type=Path, default=Path(__file__).resolve().parents[1] / 'tools')
    parser.add_argument('--stop-after', choices=['interpretation'])
    parser.add_argument('--checkpoint-attribution', action='store_true',
                        help='recompute decoder activations during RelP backward to reduce activation storage')
    parser.add_argument('--followup', action=argparse.BooleanOptionalAction, default=None,
                        help='continue with up to two follow-up rounds (default: enabled for full --config runs)')
    args = parser.parse_args()
    if args.checkpoint_attribution and (args.config is None or args.stop_after):
        parser.error('--checkpoint-attribution requires a full --config investigation')
    if args.followup and (args.config is None or args.stop_after):
        parser.error('--followup requires a full --config investigation')
    if args.stop_after and args.config is None:
        parser.error('--stop-after requires --config')
    if args.followup is None:
        args.followup = args.config is not None and args.stop_after is None
    sys.path.insert(0, str(args.backend_tools.resolve()))
    if args.anatomy_from is not None:
        from .anatomy_replay import replay
        print(replay(args.anatomy_from, args.output) / 'index.html')
        return
    if args.ablate_from is not None:
        from .ablation_run import replay
        replay(args.ablate_from, args.output)
        return
    config = read(args.config)
    source = args.output.resolve()
    if args.followup:
        followup_root = source / 'followup'
    if config.get('publish'):
        destination = (Path(config['publish']) / config['neuron']).resolve()
        if destination == source or destination in source.parents or source in destination.parents:
            parser.error('Publication must be outside the investigation directory and its ancestors')
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.stop_after == 'interpretation' else ','.join(map(str, config['gpus']))
    os.environ['NEURON_MICROSCOPE_SETTINGS'] = str(args.output / 'backend-settings.json')
    write(Path(os.environ['NEURON_MICROSCOPE_SETTINGS']), config['settings'])
    try:
        output = run(config, args.output, stop_after=args.stop_after,
                     checkpoint_attribution=args.checkpoint_attribution)
    except Exception as error:
        import traceback
        write(args.output / 'failure.json', {'at': now(), 'error': str(error), 'traceback': traceback.format_exc()})
        Timeline(args.output).status('failed', error=str(error))
        raise
    if args.followup:
        from .followups import run as run_followups
        output = run_followups(source, followup_root, in_place=True)
        print(output / 'index.html')
        if read(followup_root / 'status.json')['phase'] != 'complete':
            return 1
    if config.get('publish'):
        from comparison import publish
        destination = Path(config['publish']) / config['neuron']
        from .render import directory
        with lock(Path(config['publish']) / '.directory.lock'):
            exists = destination.exists()
            if args.stop_after is None or not exists:
                publish(output, destination, config['neuron'], config['references'], replace=exists)
            directory(config['publish'], config.get('publish_neurons', [config['neuron']]), config['references'])
    return 0
