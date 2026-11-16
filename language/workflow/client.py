"""DeepSeek requests, bounded concurrency and complete attempt logs."""
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
from . import prompts
from .state import read, write
from .contracts import validate


def parse_output(answer, schema):
    answer = answer.strip()
    if answer.startswith('```'):
        answer = answer.partition('\n')[2].lstrip()
    output, end = json.JSONDecoder().raw_decode(answer)
    tail = answer[end:].strip().removesuffix('```').strip()
    # Recover redundant delimiters or an exact duplicate without selecting conflicting output.
    if tail and (not isinstance(output, dict) or (tail != answer[:end] and tail.strip('}] \t\r\n'))):
        raise json.JSONDecodeError('Extra data', answer, end)
    if (isinstance(output, dict) and schema['type'] == 'object' and
            'type' not in schema.get('properties', {}) and output.get('type') == 'json_object'):
        del output['type']
    if isinstance(output, dict):
        for pole in ('positive', 'negative'):
            fields = schema.get('properties', {}).get(pole, {}).get('items', {}).get('properties', {})
            if {'id', 'name', 'description'} <= fields.keys() and 'predicted_pole' not in fields:
                for rule in output.get(pole, []):
                    if isinstance(rule, dict) and rule.get('predicted_pole') == pole:
                        del rule['predicted_pole']
    return output


def validation_feedback(output, schema, error):
    """Report all invalid enum values so one correction can repair the whole response."""
    invalid = []

    def visit(value, rule, path):
        if 'enum' in rule and value not in rule['enum']:
            invalid.append(f'{path}: unexpected value {value!r}')
        if isinstance(value, dict):
            for key, child in rule.get('properties', {}).items():
                if key in value:
                    visit(value[key], child, path + '.' + key)
        elif isinstance(value, list) and 'items' in rule:
            for index, item in enumerate(value):
                visit(item, rule['items'], f'{path}[{index}]')

    visit(output, schema, '$')
    if not invalid:
        return str(error)
    if str(error) not in invalid:
        invalid.insert(0, str(error))
    return '; '.join(invalid) + prompts.ENUM_CORRECTION


class Client:
    def __init__(self, config, root, slots):
        self.config, self.root, self.slots = config, Path(root), Path(slots)
        self.root.mkdir(parents=True, exist_ok=True)
        self.slots.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def slot(self):
        if self.config.get('max_concurrent_requests') is None:
            yield
            return
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

    def request_text(self, name, prompt, packet, stage):
        return self.request(name, prompt, packet, None, stage)['text']

    def request(self, name, prompt, packet, schema, stage, check=None):
        folder = self.root / name
        system = prompt if schema is None else prompt + prompts.SCHEMA_HEADER + json.dumps(schema)
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False)}]
        fingerprint = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()

        def decode(answer):
            return {'text': answer} if schema is None else parse_output(answer, schema)

        def checked(output):
            if schema is None:
                text = output['text']
                if not isinstance(text, str) or not text.strip():
                    raise ValueError('The requested prose is empty.')
                output = {'text': text.strip()}
            else:
                validate(output, schema)
            if check:
                check(output)
            return output

        def feedback_for(answer, error):
            if schema is not None:
                try:
                    return validation_feedback(decode(answer), schema, error)
                except (ValueError, KeyError, TypeError):
                    pass
            return str(error)

        def add_correction(answer, feedback):
            instruction = prompts.TEXT_CORRECTION if schema is None else prompts.JSON_CORRECTION
            if schema is not None or answer.strip():
                messages.append({'role': 'assistant', 'content': answer})
            messages.append({'role': 'user', 'content': instruction + feedback})

        if (folder / 'result.json').exists():
            saved = read(folder / 'result.json')
            if saved['input_id'] != fingerprint:
                raise ValueError(f'Inputs changed for completed step {name}; choose a new output directory')
            try:
                output = checked(saved['output'])
            except (ValueError, KeyError, TypeError):
                pass
            else:
                return output
        folder.mkdir(parents=True, exist_ok=True)
        last_invalid = None
        for attempt in sorted(folder.glob('attempt-*'), reverse=True):
            receipt = attempt / 'completion.json'
            if not receipt.is_file():
                continue
            completion = read(receipt)
            if completion.get('status') == 'failed' or completion.get('finish_reason') not in ('stop', 'end_turn'):
                continue
            request = read(attempt / 'request.json')
            if request.get('messages', request.get('input'))[:2] != messages:
                continue
            try:
                answer = (attempt / 'output.txt').read_text(encoding='utf-8').strip()
                output = checked(decode(answer))
            except (ValueError, KeyError, TypeError) as error:
                if last_invalid is None:
                    last_invalid = (answer, feedback_for(answer, error))
                continue
            write(folder / 'result.json', {'input_id': fingerprint, 'attempt': attempt.name, 'output': output})
            return output
        # A single formatting correction preserves the original answer and request.
        # Transport failures are separate from scientific feedback or iteration.
        if last_invalid is not None:
            answer, error = last_invalid
            add_correction(answer, error)
        send = self._respond if schema is None else self._stream
        attempts = 1 if last_invalid is not None else 2
        for correction in range(attempts):
            max_tokens = None
            while True:
                number = 1 + len(list(folder.glob('attempt-*')))
                destination = folder / f'attempt-{number:03d}'
                try:
                    answer = (send(messages, destination, stage) if max_tokens is None else
                              send(messages, destination, stage, max_tokens=max_tokens))
                    break
                except urllib.error.HTTPError as error:
                    available = getattr(error, 'context_output_tokens', None)
                    if max_tokens is not None or available is None or available <= 0:
                        raise
                    max_tokens = min(self.config['max_output_tokens'], available)
            try:
                output = checked(decode(answer))
                write(folder / 'result.json', {'input_id': fingerprint, 'attempt': destination.name, 'output': output})
                return output
            except (ValueError, KeyError, TypeError) as error:
                feedback = feedback_for(answer, error)
                write(destination / 'format-error.json', {'error': feedback})
                if correction == attempts - 1:
                    raise
                add_correction(answer, feedback)

    @contextlib.contextmanager
    def _exchange(self, url, request, folder, timing):
        try:
            write(folder / 'request.json', request)
            key_path = os.environ.get('NEURON_PAPER_API_KEY_FILE')
            key = Path(key_path).read_text().strip() if key_path else os.environ.get('DEEPSEEK_API_KEY', '').strip()
            if not key:
                raise ValueError('Set DEEPSEEK_API_KEY or NEURON_PAPER_API_KEY_FILE')
            ca = os.environ.get('SSL_CERT_FILE')
            if ca is None and Path('/etc/ssl/certs/ca-certificates.crt').is_file():
                ca = '/etc/ssl/certs/ca-certificates.crt'
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=ca)))
            req = urllib.request.Request(url, data=json.dumps(request, ensure_ascii=False).encode(),
                headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
            with self.slot():
                timing['start'] = time.time()
                with opener.open(req, timeout=900) as response:
                    write(folder / 'headers.json', {'status': response.status, 'request_id': response.headers.get('x-request-id')})
                    yield response
            timing['status'] = 'complete'
        except Exception as error:
            timing.update(status='failed', error=str(error))
            if isinstance(error, urllib.error.HTTPError):
                body = error.read()
                (folder / 'error-body.txt').write_bytes(body)
                if error.code == 400:
                    match = re.search(r'maximum context length is (\d+) tokens.*?\((\d+) in the messages',
                                      body.decode(errors='replace'), re.DOTALL)
                    if match:
                        error.context_output_tokens = int(match[1]) - int(match[2])
                timing['http_status'] = error.code
            raise

    def _save_attempt(self, folder, timing, content, reasoning, usage, finish, response_id):
        timing['end'] = time.time()
        started = timing.get('start', timing['end'])
        timing['queue_seconds'] = started - timing['queued_at']
        (folder / 'output.txt').write_text(content, encoding='utf-8')
        (folder / 'reasoning.txt').write_text(reasoning, encoding='utf-8')
        write(folder / 'completion.json', {'usage': usage, 'finish_reason': finish,
                                           'response_id': response_id, 'status': timing.get('status', 'failed')})
        write(folder / 'timing.json', timing)
        write(folder / 'billing.json', {'estimated_usd': price(usage, self.config['pricing'], started),
                                      'usage_recorded': bool(usage), 'usage': usage})

    def _stream(self, messages, folder, stage, max_tokens=None):
        folder.mkdir(parents=True)
        cfg = self.config
        request = {'model': cfg['model'], 'reasoning_effort': cfg['reasoning_effort'],
                   'max_tokens': cfg['max_output_tokens'] if max_tokens is None else max_tokens, 'stream': True,
                   'stream_options': {'include_usage': True}, 'response_format': {'type': 'json_object'},
                   'messages': messages}
        content, reasoning, usage, finish = [], [], {}, None
        response_id, done = None, False
        timing = {'stage': stage, 'queued_at': time.time()}
        try:
            with self._exchange(cfg['chat_url'], request, folder, timing) as response:
                with (folder / 'response.sse').open('wb') as log:
                    for line in response:
                        log.write(line)
                        log.flush()
                        if not line.startswith(b'data:'):
                            continue
                        raw = line[5:].strip()
                        if raw == b'[DONE]':
                            done = True
                            break
                        event = json.loads(raw)
                        if event.get('error'):
                            raise RuntimeError(str(event['error']))
                        identity = event.get('id')
                        if not isinstance(identity, str) or not identity:
                            raise RuntimeError('API response chunk has no completion ID')
                        if response_id is not None and identity != response_id:
                            raise RuntimeError('API response contains multiple completion IDs')
                        response_id = identity
                        if event.get('usage'):
                            usage.update(event['usage'])
                        choices = event.get('choices', [])
                        if not choices:
                            continue
                        if len(choices) != 1 or choices[0].get('index') != 0:
                            raise RuntimeError('API response must contain only choice index 0')
                        if finish is not None:
                            raise RuntimeError('API response contains choice data after its finish reason')
                        choice = choices[0]
                        delta = choice.get('delta', {})
                        c, r = delta.get('content') or '', delta.get('reasoning_content') or ''
                        if c or r:
                            timing.setdefault('first_chunk', time.time())
                            timing['last_chunk'] = time.time()
                        content.append(c)
                        reasoning.append(r)
                        finish = choice.get('finish_reason')
                if not done or not content or finish not in ('stop', 'end_turn'):
                    raise RuntimeError(f'Incomplete API response: finish_reason={finish}, done={done}')
        finally:
            self._save_attempt(folder, timing, ''.join(content), ''.join(reasoning), usage, finish, response_id)
        return ''.join(content)

    def _respond(self, messages, folder, stage, max_tokens=None):
        cfg = self.config
        endpoint = urlsplit(cfg['chat_url'])
        path = endpoint.path.rstrip('/')
        if not path.endswith('/chat/completions'):
            raise ValueError('Responses API requires a chat_url ending in /chat/completions')
        url = endpoint._replace(path=path.removesuffix('/chat/completions') + '/responses').geturl()
        request = {'model': cfg['model'], 'reasoning': {'effort': cfg['reasoning_effort']},
                   'max_output_tokens': cfg['max_output_tokens'] if max_tokens is None else max_tokens,
                   'stream': False, 'input': messages,
                   'text': {'format': {'type': 'text'}}}
        folder.mkdir(parents=True)
        timing = {'stage': stage, 'queued_at': time.time()}
        content, reasoning, usage, finish, response_id = '', '', {}, None, None
        try:
            with self._exchange(url, request, folder, timing) as response:
                raw = response.read()
                (folder / 'response.json').write_bytes(raw)
                payload = json.loads(raw)
                response_id = payload.get('id')
                if tokens := payload.get('usage'):
                    usage = {'prompt_tokens': tokens['input_tokens'], 'completion_tokens': tokens['output_tokens'],
                             'prompt_cache_hit_tokens': (tokens.get('input_tokens_details') or {}).get('cached_tokens', 0),
                             'total_tokens': tokens.get('total_tokens', tokens['input_tokens'] + tokens['output_tokens'])}
                status = payload.get('status')
                if status != 'completed':
                    detail = (payload.get('incomplete_details') or {}).get('reason') or (payload.get('error') or {}).get('code')
                    raise RuntimeError(f'Responses API status={status}, reason={detail}')
                output = payload['output']
                answers = [item for item in output if item['type'] == 'message' and item['role'] == 'assistant']
                if len(answers) != 1 or answers[0].get('status', 'completed') != 'completed':
                    raise RuntimeError('Responses API must return one completed assistant message')
                parts = answers[0]['content']
                if any(part['type'] != 'output_text' for part in parts):
                    raise RuntimeError('Responses API did not return a text answer')
                content = ''.join(part['text'] for part in parts)
                reasoning = ''.join(part['text'] for item in output if item['type'] == 'reasoning'
                                    for part in item.get('content', []) if part['type'] == 'reasoning_text')
                finish = 'stop'
        finally:
            self._save_attempt(folder, timing, content, reasoning, usage, finish, response_id)
        return content


def price(usage, pricing, started):
    timestamp = datetime.datetime.fromtimestamp(started, datetime.timezone.utc)
    peak = timestamp.weekday() in pricing.get('peak_weekdays', range(5)) and any(
        a <= timestamp.hour < b for a, b in pricing.get('peak_utc_hours', []))
    factor = pricing.get('peak_multiplier', 1) if peak else 1
    hit = usage.get('prompt_cache_hit_tokens', 0)
    missed = usage.get('prompt_cache_miss_tokens', usage.get('prompt_tokens', 0) - hit)
    return factor * (hit * pricing['cached_input_per_million'] + missed * pricing['input_per_million'] +
                     usage.get('completion_tokens', 0) * pricing['output_per_million']) / 1e6
