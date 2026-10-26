"""Plan an unmeasured response pole from one report and its existing evidence."""
import json
from pathlib import Path
from common import read,write
from evidence import prepare
from blind_tests import call

SYSTEM = """Design a controlled replacement experiment for an observed neuron response. Read the saved contexts through their activation anchors, the existing interpretation, and the completed experiments. Test the requested response pole against a plausible alternative with matched replacements in one fixed frame. A negative pole is a negative signed product, not merely a weaker positive response. If its semantics are mixed or unclear, test the leading account against that uncertainty rather than inventing a coherent opposite concept. Preserve relevant prefix context and choose the scoring location before measurement. Return one JSON object with analysis and sweeps. Each sweep has pole, title, prediction, template containing exactly one {slot}, replacements as text/label objects, and score. Use replacement_span or marker scoring; use min for the negative tail, max for the positive tail, or single for a fixed token. Write 6-10 conditions including matched alternatives. Design only the missing experiment; do not repeat any completed experiment."""

def run(report, pole, output, llm=call):
    report,output=Path(report),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    b=read(report/'paper-data.json')['bundle']
    packet=prepare(report,output/'evidence')
    cases=packet['contexts_json']['cases']
    wanted=('bottom_tokens','bottom_documents') if pole=='negative' else ('top_tokens','top_documents')
    chosen=[];counts={}
    for group in wanted:
        for c in cases:
            if group not in c['groups'] or any(x['id']==c['id'] for x in chosen):continue
            if counts.get(c['document'],0)>=2:continue
            chosen.append(c);counts[c['document']]=counts.get(c['document'],0)+1
    selected=chosen[:16]
    other='top_documents' if pole=='negative' else 'bottom_documents'
    selected += [c for c in cases if other in c['groups']][:4]
    figures=(b.get('custom') or {}).get('figures',[])
    experiments=[]
    for f in figures:
        req=report/'artifacts'/('sweeps-'+f['id'])/'tool-request.json'
        experiments.append({'request':read(req) if req.exists() else f,
                            'results':f.get('results',[])})
    search=b.get('search') or {}
    evidence={'requested_pole':pole,'current_independent_interpretation':b.get('claim'),
        'distribution':b.get('distribution'),'saved_cases':selected,
        'trace_columns':packet['contexts_json']['trace_columns'],
        'token_selectivity':packet['token_selectivity_json'],
        'completed_replacement_experiments':experiments,
        'existing_search':search,
        'task':'Design one additional replacement sweep for the missing response pole. All other experiments are already complete.'}
    messages=[{'role':'system','content':SYSTEM},
              {'role':'user','content':json.dumps(evidence,ensure_ascii=False)}]
    result=llm(messages,output/'planner','missing-pole-planner')
    write(output/'plan.json',result)
    return result
