"""Upload a checked snapshot through the Site's bounded, authenticated R2 API."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path
import threading
import fcntl
from urllib.parse import urlsplit
from .site_export import within
import time
import urllib.request
import urllib.error

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def verified_existing(records, request, workers=8):
    """Reuse only objects whose stored size and digest the server verified."""
    def check(batch):
        result = request('/api/import/verify', {'objects': batch})
        keys = {r['key'] for r in batch}
        bad = set(result['bad'])
        if result['checked'] != len(batch) or not bad.issubset(keys):
            raise ValueError('Invalid existing-object verification receipt')
        return [r for r in batch if r['key'] not in bad]
    batches = [records[i:i+100] for i in range(0, len(records), 100)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [record for batch in pool.map(check, batches) for record in batch]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--origin', required=True, help='HTTPS origin of your own deployment')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--available-only', action='store_true', help='Upload files already copied; never activate an incomplete release.')
    p.add_argument('--manifest', default='upload-manifest.json')
    p.add_argument('--reuse-existing', action='store_true', help='Verify cloud objects and skip already stored identical content.')
    args = p.parse_args(argv)
    if not 1 <= args.workers <= 32:p.error('--workers must be between 1 and 32')
    parsed=urlsplit(args.origin)
    if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.path not in ('','/') or parsed.query or parsed.fragment:
        p.error('--origin must be an HTTPS origin without credentials, path or query')
    origin=args.origin.rstrip('/')
    lock=(args.root/'.publication.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest = json.loads((args.root / args.manifest).read_bytes())
    records = []
    # Validate every local object before sending any publication request.
    for item in manifest['objects']:
        path = within(args.root,item['key'])
        if args.available_only and not path.is_file():
            continue
        raw = path.read_bytes()
        valid = len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
        if args.available_only and not valid:
            continue  # A previous interrupted copy may have the right size but wrong bytes.
        assert valid, item['key']
        records.append(item)
    credentials = json.loads(getpass.getpass('Temporary credentials JSON (import; optional service): '))
    token = credentials['import']
    service_token = credentials.get('service')
    if len(token) != 64:
        raise ValueError('Expected temporary 64-character token')
    local = threading.local()
    stopped = threading.Event()
    def request(route, body):
        if stopped.is_set():
            raise RuntimeError('Upload stopped after an earlier failure')
        if not hasattr(local, 'session'):
            local.session = urllib.request.build_opener(NoRedirect())
        for attempt in range(6):
            try:
                req = urllib.request.Request(origin + route, data=json.dumps(body).encode(),
                    headers={'X-Microscope-Import': token, **({'OAI-Sites-Authorization': 'Bearer ' + service_token} if service_token else {}), 'Content-Type': 'application/json'})
                with local.session.open(req, timeout=60) as response:
                    return json.load(response)
            except urllib.error.HTTPError as r:
                if r.code in (401, 403):
                    stopped.set()
                    raise PermissionError(f'Import authorization rejected: HTTP {r.code}: {r.read(160).decode()}')
                if r.code not in (408, 429, 500, 502, 503, 504):
                    stopped.set()
                    detail = r.read().decode()
                    try:
                        failures = [v for v in json.loads(detail).get('results', []) if v.get('status') != 200]
                        if failures:
                            detail = json.dumps(failures[:5])
                    except ValueError:
                        pass
                    raise ValueError(f'Import request rejected: HTTP {r.code}: {detail[:1000]}')
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 5:
                    raise
            time.sleep(min(2 ** attempt, 15))
        stopped.set()
        raise RuntimeError('Upload retry limit reached')
    assert request('/api/import/verify', {'objects': []}) == {'checked': 0, 'bad': []}
    ack_file = args.root / 'upload-ack.jsonl'
    if args.reuse_existing:
        print(json.dumps({'phase': 'check_existing', 'objects': len(records)}), flush=True)
        existing = verified_existing(records, request, workers=min(args.workers,16))
        with ack_file.open('a') as log:
            for record in existing:
                log.write(json.dumps({**record, 'origin': origin}) + '\n')
        print(json.dumps({'phase': 'reused', 'objects': len(existing)}), flush=True)
    acknowledged = set()
    if ack_file.exists():
        for line in ack_file.read_text().splitlines():
            try:
                record = json.loads(line)
                if record.get('origin') == origin:
                    acknowledged.add((record['key'], record['sha256']))
            except (ValueError, KeyError):
                pass
    pending = [r for r in records if (r['key'], r['sha256']) not in acknowledged]
    batches, batch, length = [], [], 0
    for record in pending:
        estimated = record['bytes'] * 4 // 3 + 512
        if batch and (len(batch) >= 16 or length + estimated > 3 * 1024 * 1024):
            batches.append(batch)
            batch, length = [], 0
        batch.append(record)
        length += estimated
    if batch:
        batches.append(batch)
    started = time.monotonic()
    print(json.dumps({'phase': 'upload', 'objects': len(records), 'pending': len(pending), 'batches': len(batches)}), flush=True)
    def upload(batch):
        objects = [{**r, 'base64': base64.b64encode(within(args.root,r['key']).read_bytes()).decode()} for r in batch]
        reply = request('/api/import/batch', {'objects': objects})
        expected = {(r['key'], r['sha256'], r['bytes']) for r in batch}
        actual = {(r['key'], r['sha256'], r['bytes']) for r in reply['results'] if r['status'] == 200}
        assert actual == expected, 'Incomplete batch acknowledgement'
        return batch
    uploaded, byte_count = 0, 0
    with ack_file.open('a', buffering=1) as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(upload, b) for b in batches]
        for index, future in enumerate(as_completed(futures)):
            for record in future.result():
                log.write(json.dumps({**record,'origin':origin}) + '\n')
                uploaded += 1
                byte_count += record['bytes']
            if (index + 1) % 20 == 0 or index + 1 == len(futures):
                print(json.dumps({'phase': 'upload', 'uploaded': uploaded, 'pending': len(pending),
                    'MiB': round(byte_count / 2**20, 1), 'seconds': round(time.monotonic() - started, 1)}), flush=True)
    # Independently HEAD-check every stored object before atomically switching active.json.
    print(json.dumps({'phase': 'verify', 'objects': len(records)}), flush=True)
    def verify(batch):
        result = request('/api/import/verify', {'objects': batch})
        assert result['checked'] == len(batch) and result['bad'] == [], result
    with ThreadPoolExecutor(max_workers=min(args.workers,16)) as pool:
        list(pool.map(verify, [records[i:i+100] for i in range(0, len(records), 100)]))
    if len(records) != len(manifest['objects']):
        print(json.dumps({'phase': 'partial_upload_verified', 'objects': len(records), 'total': len(manifest['objects'])}), flush=True)
        return
    release = manifest['summary']['release']
    examples = manifest['summary'].get('kind') == 'examples'
    prefix = 'examples' if examples else 'releases'
    m = next(r for r in records if r['key'] == f'{prefix}/{release}/manifest.json')
    active = request('/api/import/publish-examples' if examples else '/api/import/publish', {'release': release, 'manifestSha256': m['sha256']})
    count = 'samples' if examples else 'variants'
    assert active['release'] == release and active[count] == manifest['summary'][count]
    receipt = {'origin':origin, 'published': active, 'verifiedObjects': len(records), 'verifiedBytes': sum(r['bytes'] for r in records),
        'completedAt': datetime.now(timezone.utc).isoformat()}
    (args.root / 'publication.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'phase': 'published', **receipt}), flush=True)

if __name__ == '__main__':
    main()
