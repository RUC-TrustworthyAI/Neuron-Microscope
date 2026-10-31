#!/usr/bin/env python3
"""Synchronize a prepared public walkthrough and its saved report assets."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
FILES = ('index.html', 'replay.js', 'i18n.js', 'replay.css', 'demo.css')
DIRECTORIES = ('content', 'reports', 'screenshots')


def export(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError('Use separate source and output directories')
    for name in FILES:
        if not (source / name).is_file():
            raise ValueError(f'The prepared walkthrough is missing {name}')
    for name in DIRECTORIES:
        if not (source / name).is_dir():
            raise ValueError(f'The prepared walkthrough is missing {name}/')
    episode = json.loads((source / 'content/episode.json').read_text())
    required = {'snapshots', 'stages', 'preparation', 'predictions', 'measurements', 'costs', 'aux_calls'}
    if not required.issubset(episode):
        raise ValueError('The prepared episode must match the walkthrough data format')

    assets = {Path(name): source / name for name in FILES}
    for name in DIRECTORIES:
        for path in (source / name).rglob('*'):
            if path.is_symlink():
                raise ValueError('Walkthrough assets must be regular files')
            if path.is_file():
                assets[path.relative_to(source)] = path
    # These directories contain only the prepared recording's display assets.
    for name in DIRECTORIES:
        folder = target / name
        if folder.exists():
            for path in folder.rglob('*'):
                if path.is_file() and path.relative_to(target) not in assets:
                    path.unlink()
    for relative, original in sorted(assets.items()):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
    print(f'Synchronized {len(assets)} public walkthrough assets to {target}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Prepared public walkthrough with content, reports and screenshots')
    parser.add_argument('--output', type=Path, default=ROOT / 'website')
    args = parser.parse_args()
    export(args.source, args.output)
