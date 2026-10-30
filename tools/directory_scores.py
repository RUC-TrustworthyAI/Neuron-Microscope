"""Compare completed explanation texts for the publication directory."""
import concurrent.futures
import http.server
import json
import re
import threading
import time
import urllib.request
from pathlib import Path
from common import write

PROMPT = Path(__file__).with_name('comparison_assets') / 'similarity-prompt.txt'


def score_rows(rows, records, manifest=None):
    """Score new pairs with two workers; retain requests, streams and judgments."""
    from api_meter import Budget, handler
    records = Path(records)
    comparisons = records / 'comparisons'
    comparisons.mkdir(parents=True, exist_ok=True)
    pending = []
    for row in rows:
        if 'input' not in row or row.get('judgment'):
            continue
        attempt = comparisons / (row['neuron'] + '-input.json')
        if attempt.exists() and json.loads(attempt.read_text()) == row['input']:
            continue
        pending.append(row)
    if not pending:
        return
    write(records / 'manifest.json', manifest if manifest is not None else {'budget_usd': None})
    budget = Budget(records)
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler(budget))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    llm = getattr(budget, 'llm', {'model': 'deepseek-flash', 'reasoning_effort': 'high'})

    def score(row):
        neuron = row['neuron']
        write(comparisons / (neuron + '-input.json'), row['input'])
        request = {'model': llm['model'], 'reasoning_effort': llm['reasoning_effort'], 'stream': True,
                   'messages': [{'role': 'system', 'content': PROMPT.read_text()},
                                {'role': 'user', 'content': json.dumps(row['input'], ensure_ascii=False)}]}
        write(comparisons / (neuron + '-request.json'), request)
        req = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                                     data=json.dumps(request).encode(),
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + budget.token})
        content, reasoning = [], []
        model, usage = None, None
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=900) as response:
                for line in response:
                    if not line.startswith(b'data:'):
                        continue
                    data = line[5:].strip()
                    if not data or data == b'[DONE]':
                        continue
                    event = json.loads(data)
                    model = event.get('model', model)
                    usage = event.get('usage') or usage
                    for choice in event.get('choices', []):
                        delta = choice.get('delta', {})
                        content.append(delta.get('content') or '')
                        reasoning.append(delta.get('reasoning_content') or '')
            text = ''.join(content)
            write(comparisons / (neuron + '-response.json'),
                  {'model': model, 'content': text, 'reasoning_content': ''.join(reasoning),
                   'usage': usage, 'elapsed_seconds': time.monotonic() - started})
            judgment = json.loads(re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', text))
            if type(judgment['semantic_similarity']) is not int or not 1 <= judgment['semantic_similarity'] <= 5:
                raise ValueError('Similarity score must be an integer from 1 to 5')
            if not isinstance(judgment['reason'], str) or not judgment['reason'].strip():
                raise ValueError('Similarity judgment requires a rationale')
            write(comparisons / (neuron + '-judgment.json'),
                  {'input': row['input'], 'comparison_a': row['comparison_a'], 'model': model, 'judgment': judgment})
            row['judgment'] = judgment
        except Exception as error:
            write(comparisons / (neuron + '-error.json'), {'error': str(error)})

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(score, pending))
    finally:
        server.shutdown()
        server.server_close()
