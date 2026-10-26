import dataclasses,json,os,time
from pathlib import Path
from deepseek_harness import DeepSeekHarness
os.environ['PATH']='/control/bin:'+os.environ.get('PATH','/usr/local/bin:/usr/bin:/bin')
p=Path('/logs/agent');p.mkdir(parents=True,exist_ok=True)
config=json.loads(Path('/control/llm.json').read_text())
with (p/'events.jsonl').open('a') as log:
 def event(e):
  obj=dataclasses.asdict(e) if dataclasses.is_dataclass(e) else e
  log.write(json.dumps(obj,ensure_ascii=False,default=str)+'\n');log.flush()
 try:
  with DeepSeekHarness(dsh_home='/workspace/.dsh',cwd='/workspace',profile='sdk-minimal',
      patches=('/control/output-policy.yml',),
      provider=config.get('provider','deepseek-official'),model=config.get('model','deepseek-flash'),reasoning_effort=config.get('reasoning_effort','high'),
      max_tokens=config.get('max_output_tokens',393216),base_url=os.environ['DEEPSEEK_BASE_URL'],
      api_key=os.environ['DEEPSEEK_API_KEY']) as harness:
   result=harness.run(Path('/control/task.txt').read_text(),session_id='neuron-paper',on_notification=event)
  (p/'final-response.txt').write_text(result.final_response)
  (p/'AGENT_TERMINAL.json').write_text(json.dumps({'finish_reason':result.finish_reason,'finished_at':time.time()}))
 except BaseException as e:
  (p/'AGENT_TERMINAL.json').write_text(json.dumps({'error_type':type(e).__name__,'error':str(e),'finished_at':time.time()}))
  raise
