"""Shared native-Messages/ChatCompletions meter with the existing configured tariff."""
import datetime as dt
import http.server,json,os,secrets,threading,urllib.request,urllib.error,ssl,time,hashlib
from pathlib import Path
from common import write as save
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);save(path,value)

class Budget:
    def __init__(self,run):
        self.run=run;self.lock=threading.Lock();self.spent=0.;self.pending=0.;self.unknown=0.;self.calls=0
        (run/'logs/api').mkdir(parents=True,exist_ok=True)
        manifest=json.loads((run/'manifest.json').read_text())
        limit=manifest.get('budget_usd',2)
        from settings import load
        defaults=load()
        self.llm={**defaults['llm'],**manifest.get('settings',{}).get('llm',{})}
        self.pricing={**defaults['pricing'],**manifest.get('settings',{}).get('pricing',{})}
        self.limit=None if limit is None else float(limit)
        self.token=secrets.token_urlsafe(32)
        key_file=os.environ.get('NEURON_PAPER_API_KEY_FILE')
        self.key=Path(key_file).read_text().strip() if key_file else os.environ.get('DEEPSEEK_API_KEY','').strip()
        if not self.key:raise ValueError('Set DEEPSEEK_API_KEY or NEURON_PAPER_API_KEY_FILE on the host.')
        from api_concurrency import RequestSlots
        from settings import paths
        state=Path(manifest.get('paths',{}).get('state') or paths(defaults)['state'])
        account=hashlib.sha256(self.key.encode()).hexdigest()
        self.slots=RequestSlots(state/'api-concurrency'/account,self.llm.get('max_concurrent_requests',10))
        self.usage={'prompt_tokens':0,'prompt_cache_hit_tokens':0,'prompt_cache_miss_tokens':0,'completion_tokens':0,'total_tokens':0}
        previous=run/'logs/usage.json'
        if previous.exists():
            prior=json.loads(previous.read_text());self.spent=prior['estimated_usd'];self.unknown=prior.get('unknown_charge_upper_usd',0)+prior.get('reserved_usd',0);self.calls=prior['requests']
            self.usage={k:prior.get(k,0) for k in self.usage}
        self.persist()
    def persist(self):
        write(self.run/'logs/usage.json',{**self.usage,'requests':self.calls,'estimated_usd':self.spent,
             'reserved_usd':self.pending,'unknown_charge_upper_usd':self.unknown,'budget_usd':self.limit,
             'note':'Token-price estimate, not account invoice. Admission reserves peak-rate upper bounds.'})


def handler(budget):
 class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        self.send_response(200);self.end_headers();self.wfile.write(b'{"status":"ready"}')
    def do_POST(self):
        if self.headers.get('Authorization')!='Bearer '+budget.token and self.headers.get('x-api-key')!=budget.token:
            self.send_error(403);return
        raw=self.rfile.read(int(self.headers.get('Content-Length','0')))
        try:req=json.loads(raw)
        except Exception:self.send_error(400);return
        is_messages=self.path.rstrip('/').endswith('/messages')
        req['model']=budget.llm['model']
        if not is_messages:req['reasoning_effort']=budget.llm['reasoning_effort']
        # Provider maximum; no lower application-level output cap.
        req['max_tokens']=budget.llm['max_output_tokens']
        if req.get('stream') and not is_messages:req['stream_options']={'include_usage':True}
        raw=json.dumps(req,ensure_ascii=False,separators=(',',':')).encode()
        prices=budget.pricing
        peak_factor=max(1,prices['peak_multiplier'])
        reserve=(min(len(raw)*1.05+4096,1_000_000)*prices['input_per_million']+
                 req['max_tokens']*prices['output_per_million'])*peak_factor/1e6
        while True:
            with budget.lock:
                if budget.limit is not None and budget.spent+budget.unknown+reserve>budget.limit:
                    self.send_response(402);self.end_headers();self.wfile.write(b'{"error":{"message":"Run budget exhausted","type":"budget_limit"}}');return
                if budget.limit is None or budget.spent+budget.pending+budget.unknown+reserve<=budget.limit:
                    budget.calls+=1;number=budget.calls;budget.pending+=reserve;budget.persist()
                    break
            time.sleep(.25)
        dest=budget.run/'logs/api'/f'{number:04d}';dest.mkdir()
        write(dest/'request.json',req)
        started=dt.datetime.now(dt.timezone.utc)
        write(dest/'attempt.json',{'started_at':started.isoformat(),'reserved_usd':reserve,'client_route':self.path})
        usage=None;attempted=False;completed=False;slot=None;queue_seconds=0.
        first_chunk=None;last_chunk=None
        request_clock=time.monotonic();first_chunk_clock=None;last_chunk_clock=None
        try:
            slot=budget.slots.acquire();queue_seconds=time.monotonic()-request_clock
            endpoint=budget.llm['messages_url'] if is_messages else budget.llm['chat_url']
            headers={'x-api-key':budget.key,'anthropic-version':'2023-06-01','Content-Type':'application/json'} if is_messages else {'Authorization':'Bearer '+budget.key,'Content-Type':'application/json'}
            request=urllib.request.Request(endpoint,data=raw,headers=headers)
            ca=os.environ.get('SSL_CERT_FILE')
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=ca)))
            attempted=True
            with opener.open(request,timeout=900) as resp, (dest/'response.sse').open('wb') as log:
                self.send_response(resp.status);self.send_header('Content-Type',resp.headers.get('Content-Type','text/event-stream'));self.end_headers()
                write(dest/'headers.json',{'status':resp.status,'date':resp.headers.get('Date'),'request_id':resp.headers.get('x-request-id')})
                for line in resp:
                    if line.startswith(b'data:'):
                        last_chunk=dt.datetime.now(dt.timezone.utc);last_chunk_clock=time.monotonic()
                        if first_chunk is None:first_chunk=last_chunk;first_chunk_clock=last_chunk_clock
                    log.write(line);log.flush()
                    if line.startswith(b'data:'):
                        try:
                            event=json.loads(line[5:].strip())
                            event_usage=event.get('message',{}).get('usage') or event.get('usage')
                            if event_usage:usage={**(usage or {}),**event_usage}
                        except (ValueError,AttributeError):pass
                    try:self.wfile.write(line);self.wfile.flush()
                    except (BrokenPipeError,ConnectionResetError):pass
                completed=True
        except urllib.error.HTTPError as e:
            body=e.read()
            (dest/'error-body.json').write_bytes(body)
            rejected=400<=e.code<500 and e.code!=408
            write(dest/'error.json',{'type':'HTTPError','error':str(e),'status':e.code,
                  'http_request_sent':True,'rejected_without_generation':rejected})
            if rejected:attempted=False
            try:
                self.send_response(e.code);self.send_header('Content-Type','application/json');self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        except Exception as e:
            pre_http_tls_failure=isinstance(e,urllib.error.URLError) and isinstance(getattr(e,'reason',None),ssl.SSLCertVerificationError)
            write(dest/'error.json',{'type':type(e).__name__,'error':str(e),'http_request_sent':False if pre_http_tls_failure else None})
            if pre_http_tls_failure:attempted=False
            try:self.send_error(502,'Upstream request failed')
            except Exception:pass
        finally:
            if slot is not None:budget.slots.release(slot)
            write(dest/'timing.json',{'started_at':started.isoformat(),
                'first_chunk_at':first_chunk.isoformat() if first_chunk else None,
                'last_chunk_at':last_chunk.isoformat() if last_chunk else None,
                'finished_at':dt.datetime.now(dt.timezone.utc).isoformat(),
                'request_seconds':time.monotonic()-request_clock,'queue_seconds':queue_seconds,
                'stream_seconds':last_chunk_clock-first_chunk_clock if first_chunk_clock is not None else 0.,
                'completed':completed})
            with budget.lock:
                budget.pending-=reserve
                if usage:
                    raw_usage=dict(usage)
                    if is_messages:
                        hit=int(usage.get('cache_read_input_tokens',0));miss=int(usage.get('input_tokens',0))+int(usage.get('cache_creation_input_tokens',0))
                        usage={**usage,'prompt_tokens':hit+miss,'prompt_cache_hit_tokens':hit,'prompt_cache_miss_tokens':miss,'completion_tokens':int(usage.get('output_tokens',0)),'total_tokens':hit+miss+int(usage.get('output_tokens',0))}
                    hit=usage.get('prompt_cache_hit_tokens',usage.get('prompt_tokens_details',{}).get('cached_tokens',0))
                    miss=usage.get('prompt_cache_miss_tokens',usage['prompt_tokens']-hit)
                    peak=started.weekday() in prices['peak_weekdays'] and any(a<=started.hour<b for a,b in prices['peak_utc_hours'])
                    factor=prices['peak_multiplier'] if peak else 1
                    cost=factor*(hit*prices['cached_input_per_million']+miss*prices['input_per_million']+usage['completion_tokens']*prices['output_per_million'])/1e6
                    budget.spent+=cost
                    normalized={**usage,'prompt_cache_hit_tokens':hit,'prompt_cache_miss_tokens':miss}
                    for k in budget.usage:budget.usage[k]+=normalized.get(k,0)
                    write(dest/'usage-raw.json',raw_usage)
                    write(dest/'usage.json',usage)
                    write(dest/'billing.json',{'estimated_usd':cost,'peak_at_request_start':peak,'completed':completed,'pricing':prices})
                elif attempted:budget.unknown+=reserve
                budget.persist()
 return Handler
