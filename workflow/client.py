"""Direct streaming Chat Completions calls, bounded concurrency and complete logs."""
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from .state import read, write
from .contracts import validate


class Client:
    def __init__(self, config, root, slots):
        self.config, self.root, self.slots = config, Path(root), Path(slots)
        self.root.mkdir(parents=True, exist_ok=True)
        self.slots.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def slot(self):
        while True:
            for i in range(self.config['max_concurrent_requests']):
                stream = (self.slots / f'{i}.lock').open('a+')
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                    continue
                try:
                    yield
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)
                    stream.close()
                return
            time.sleep(.2)

    def request(self, name, prompt, packet, schema, stage, check=None):
        folder = self.root / name
        messages = [{'role': 'system', 'content': prompt + '\nJSON schema:\n' + json.dumps(schema)},
                    {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False)}]
        fingerprint = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        if (folder / 'result.json').exists():
            saved = read(folder / 'result.json')
            if saved['input_id'] != fingerprint:
                raise ValueError(f'Inputs changed for completed step {name}; choose a new output directory')
            return saved['output']
        folder.mkdir(parents=True, exist_ok=True)
        last_invalid = None
        for attempt in sorted(folder.glob('attempt-*'), reverse=True):
            completion = attempt / 'completion.json'
            if not completion.exists() or read(completion).get('finish_reason') not in ('stop', 'end_turn'):
                continue
            if read(attempt / 'request.json')['messages'][:2] != messages:
                continue
            try:
                answer = (attempt / 'output.txt').read_text().strip()
                if answer.startswith('```'):
                    answer = answer.split('\n', 1)[1].rsplit('```', 1)[0]
                output = json.loads(answer)
                validate(output, schema)
                if check:
                    check(output)
            except (ValueError, KeyError, TypeError) as error:
                if last_invalid is None:
                    last_invalid = (answer, str(error))
                continue
            write(folder / 'result.json', {'input_id': fingerprint, 'output': output})
            return output
        # A single formatting correction preserves the original answer and request.
        # Transport failures are separate from scientific feedback or iteration.
        if last_invalid is not None:
            answer, error = last_invalid
            messages += [{'role': 'assistant', 'content': answer},
                         {'role': 'user', 'content': 'Correct the JSON contract error while preserving the scientific content: ' + error}]
        attempts = 1 if last_invalid is not None else 2
        for correction in range(attempts):
            number = 1 + len(list(folder.glob('attempt-*')))
            destination = folder / f'attempt-{number:03d}'
            answer = self._stream(messages, destination, stage)
            try:
                stripped = answer.strip()
                if stripped.startswith('```'):
                    stripped = stripped.split('\n', 1)[1].rsplit('```', 1)[0]
                output = json.loads(stripped)
                validate(output, schema)
                if check:
                    check(output)
                write(folder / 'result.json', {'input_id': fingerprint, 'output': output})
                return output
            except (ValueError, KeyError, TypeError) as error:
                write(destination / 'format-error.json', {'error': str(error)})
                if correction == attempts - 1:
                    raise
                messages += [{'role': 'assistant', 'content': answer},
                             {'role': 'user', 'content': 'Correct the JSON contract error while preserving the scientific content: ' + str(error)}]

    def _stream(self, messages, folder, stage):
        folder.mkdir(parents=True)
        cfg = self.config
        request = {'model': cfg['model'], 'reasoning_effort': cfg['reasoning_effort'],
                   'max_tokens': cfg['max_output_tokens'], 'stream': True,
                   'stream_options': {'include_usage': True}, 'response_format': {'type': 'json_object'},
                   'messages': messages}
        write(folder / 'request.json', request)
        key_path = os.environ.get('NEURON_PAPER_API_KEY_FILE')
        key = Path(key_path).read_text().strip() if key_path else os.environ.get('DEEPSEEK_API_KEY', '').strip()
        if not key:
            raise ValueError('Set DEEPSEEK_API_KEY or NEURON_PAPER_API_KEY_FILE')
        content, reasoning, usage, finish = [], [], {}, None
        started = time.time()
        timing = {'stage': stage, 'queued_at': started}
        ca = os.environ.get('SSL_CERT_FILE')
        if ca is None and Path('/etc/ssl/certs/ca-certificates.crt').is_file():
            ca = '/etc/ssl/certs/ca-certificates.crt'
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=ca)))
        try:
            with self.slot():
                timing['start'] = time.time()
                req = urllib.request.Request(cfg['chat_url'], data=json.dumps(request, ensure_ascii=False).encode(),
                    headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
                with opener.open(req, timeout=900) as response, (folder / 'response.sse').open('wb') as log:
                    write(folder / 'headers.json', {'status': response.status, 'request_id': response.headers.get('x-request-id')})
                    for line in response:
                        log.write(line)
                        log.flush()
                        if not line.startswith(b'data:'):
                            continue
                        raw = line[5:].strip()
                        if raw == b'[DONE]':
                            continue
                        event = json.loads(raw)
                        if event.get('error'):
                            raise RuntimeError(str(event['error']))
                        if event.get('usage'):
                            usage.update(event['usage'])
                        for choice in event.get('choices', []):
                            delta = choice.get('delta', {})
                            c, r = delta.get('content') or '', delta.get('reasoning_content') or ''
                            if c or r:
                                timing.setdefault('first_chunk', time.time())
                                timing['last_chunk'] = time.time()
                            content.append(c)
                            reasoning.append(r)
                            finish = choice.get('finish_reason') or finish
                if not content or finish not in ('stop', 'end_turn'):
                    raise RuntimeError(f'Incomplete API response: {finish}')
                timing['status'] = 'complete'
        except Exception as error:
            timing.update(status='failed', error=str(error))
            if isinstance(error, urllib.error.HTTPError):
                (folder / 'error-body.txt').write_bytes(error.read())
                timing['http_status'] = error.code
            raise
        finally:
            timing['end'] = time.time()
            timing['queue_seconds'] = timing.get('start', timing['end']) - started
            (folder / 'output.txt').write_text(''.join(content))
            (folder / 'reasoning.txt').write_text(''.join(reasoning))
            write(folder / 'completion.json', {'usage': usage, 'finish_reason': finish})
            write(folder / 'timing.json', timing)
            write(folder / 'billing.json', {'estimated_usd': price(usage, cfg['pricing'], timing.get('start', started)),
                                          'usage_recorded': bool(usage), 'usage': usage})
        return ''.join(content)


def price(usage, pricing, started):
    timestamp = datetime.datetime.fromtimestamp(started, datetime.timezone.utc)
    peak = timestamp.weekday() in pricing.get('peak_weekdays', range(5)) and any(
        a <= timestamp.hour < b for a, b in pricing.get('peak_utc_hours', []))
    factor = pricing.get('peak_multiplier', 1) if peak else 1
    hit = usage.get('prompt_cache_hit_tokens', 0)
    missed = usage.get('prompt_cache_miss_tokens', usage.get('prompt_tokens', 0) - hit)
    return factor * (hit * pricing['cached_input_per_million'] + missed * pricing['input_per_million'] +
                     usage.get('completion_tokens', 0) * pricing['output_per_million']) / 1e6
