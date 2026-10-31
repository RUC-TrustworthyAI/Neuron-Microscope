"""Initialize a new campaign while preserving validated published variants."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

from . import full_feature_campaign as campaign
from .all_neuron_examples import LAYERS
from .five_variants import SLOTS, write_json


def exclude_published(root, catalog_path):
    root, catalog_path = Path(root), Path(catalog_path)
    raw = catalog_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    catalog = json.loads(raw)
    dimensions = {name: channels for name, channels, _ in LAYERS}
    known = set()
    for target, flags in catalog['nodes'].items():
        layer, channel = target.split(':')
        channel = int(channel)
        if layer not in dimensions or not 0 <= channel < dimensions[layer]:
            raise ValueError('Unknown published target: ' + target)
        if len(flags) != len(SLOTS) or any(flag not in '01q?' for flag in flags):
            raise ValueError('Invalid published mode flags: ' + target)
        # q means failed quality validation, not a finished explanation.
        known.update((layer, channel, slot) for slot, flag in zip(SLOTS, flags) if flag == '1')

    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        con = campaign.database(root)
        try:
            con.execute('CREATE TABLE IF NOT EXISTS publication_resume (digest TEXT PRIMARY KEY, report TEXT NOT NULL)')
            con.commit()
            con.execute('BEGIN IMMEDIATE')
            previous = con.execute('SELECT digest, report FROM publication_resume').fetchone()
            if previous:
                if previous['digest'] != digest:
                    raise ValueError('Refusing to change the published baseline of an initialized campaign')
                report = json.loads(previous['report'])
            else:
                rows = con.execute('SELECT * FROM jobs ORDER BY rowid').fetchall()
                if any(row['state'] not in ('pending', 'unavailable') for row in rows):
                    raise ValueError('Publication filtering requires a fresh, unstarted queue')
                preserved, pending, pending_images, missing_neurons = 0, 0, 0, set()
                for row in rows:
                    channels = json.loads(row['channels'])
                    complete = [ch for ch in channels if (row['layer'], ch, row['slot']) in known]
                    remaining = [ch for ch in channels if ch not in complete]
                    preserved += len(complete)
                    pending += len(remaining)
                    pending_images += len(remaining) * (4 if row['slot'] == 'diversity' else 1)
                    missing_neurons.update((row['layer'], ch) for ch in remaining)
                    if not complete:
                        continue
                    if not remaining:
                        con.execute("UPDATE jobs SET state='published',pid=NULL,error=NULL WHERE id=?", (row['id'],))
                    else:
                        con.execute("UPDATE jobs SET state='split',pid=NULL WHERE id=?", (row['id'],))
                        for suffix, subset, state, error in [
                            ('published', complete, 'published', None),
                            ('missing', remaining, row['state'], row['error']),
                        ]:
                            con.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)',
                                        (row['id'] + '_' + suffix, row['layer'], row['slot'],
                                         json.dumps(subset), state, None, error, time.time()))
                report = {'release': catalog['release'], 'catalog_sha256': digest,
                          'published_variants_preserved': preserved, 'missing_variants': pending,
                          'missing_images': pending_images, 'missing_neurons': len(missing_neurons),
                          'quality_policy': 'Only 1 is complete; q, ?, and absent modes remain pending.',
                          'method': 'Original A optimizer and configuration unchanged; only queue selection differs.'}
                con.execute('INSERT INTO publication_resume VALUES (?,?)', (digest, json.dumps(report)))
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()
        protocol_path = root / 'protocol.json'
        protocol = json.loads(protocol_path.read_text())
        protocol['resume_from_publication'] = report
        write_json(protocol_path, protocol)
        write_json(root / 'resume-selection.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--batch-images', type=int, default=8)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    campaign.initialize(args.root, args.batch_images)
    print(json.dumps(exclude_published(args.root, args.catalog)), flush=True)
    print(json.dumps(campaign.status(args.root)), flush=True)


if __name__ == '__main__':
    main()
