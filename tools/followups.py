"""Measure each registered full-prefix condition once: local RelP and matched ablations."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from common import read, write, now
from settings import request_settings
from model_tools import record_trace
from paper_sections import readers, aggregate, factorize, LOCAL_METHOD
from relp import direct

def collect(sources, report, cfg):
    data=read(Path(report)/'report.json')
    items=[]
    def add(row, kind, source, split='probe', attribution=None):
        if row.get('status','complete')!='complete' or not row.get('selected'):return
        selected=row['selected'];pos=selected['position'];ids=list(row['token_ids'])
        document=row.get('source_document');original_position=row.get('source_position')
        if document is not None and original_position is not None:
            doc=data['documents'][str(document)]
            ids=doc['token_ids'];pos=original_position
        following=ids[pos+1] if pos+1<len(ids) else row.get('following_token_id')
        full_ids=ids[:pos+1]
        source={**source,'score':row.get('score'),'scored_positions':row.get('scored_positions'),
                'source_product':selected['product'],'following_prediction_target_id':following}
        items.append({'token_ids':full_ids,'position':pos,'following_token_id':following,
            'case_id':row.get('case_id'),'source_document':document,'source_position':original_position,
            'source_product':row.get('original_product',selected['product']),
            'split':split,'kind':kind,'score':row.get('score'),
            'source':source,'reuse_attribution':attribution
                if row.get('source_start',0)==0 and list(row['token_ids'])==full_ids else None})
    for key,count,kind in [('top_documents',cfg['followups']['natural_top'],'corpus:top'),
                           ('bottom_documents',cfg['followups']['natural_bottom'],'corpus:bottom')]:
        for identity in data['examples'].get(key,[])[:count]:
            point=data['points'][str(identity)];doc=data['documents'][point['document']]
            add({'case_id':str(identity),'token_ids':doc['token_ids'],
                 'selected':{'position':point['position'],'product':point['values'][2]},
                 'source_document':point['document'],'source_position':point['position'],
                 'score':{'location':'token_position','position':point['position'],'reduction':'single'}},
                kind,{'folder':str(report),'role':'natural_reference'},split='reference')
    for entry in sources:
        folder=Path(entry['output'])
        if (folder/'search-raw.json').exists():
            for region in read(folder/'search-raw.json'):
                for role in ['candidates','controls']:
                    for row in region[role]:
                        add(row,'search:'+region['id']+':'+role,
                            {'folder':str(folder),'region':region['id'],'role':role})
        if (folder/'conditions.json').exists():
            for row in read(folder/'conditions.json'):
                add(row,'replacement:'+entry['id'],
                    {'folder':str(folder),'condition':row['condition'],'role':'replacement'})
        if (folder/'atlas-raw.json').exists():
            for row in read(folder/'atlas-raw.json'):
                kind='group:'+row['semantic_group'] if row.get('semantic_group') else 'corpus:atlas'
                add(row,kind,{'folder':str(folder),'role':'natural'},row.get('split','train') if entry.get('settings_match') else 'reference',row if entry.get('settings_match') else None)
        if entry['tool']=='attribute' and (folder/'result.json').exists():
            result=read(folder/'result.json')
            if result.get('status')!='complete':continue
            req=read(folder/'tool-request.json');pos=result['graph']['position']
            row={**req,'token_ids':result['token_ids'],'selected':result['trace'][pos],
                 'status':'complete'}
            add(row,'graph:'+entry['id'],{'folder':str(folder),'role':'illustrated_graph'})
    return items

def identity(item, request, cfg):
    payload={'policy':'full-prefix-condition-v1','model':request['model'],
             'neuron':request['neuron'],'settings':cfg,'model_identity':request.get('model_identity'),
             'versions':request.get('versions'),'token_ids':item['token_ids'],
             'position':item['position']}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def measure(runtime, item, units, controls, output, cfg):
    torch=runtime.torch;ids=item['token_ids'];pos=item['position']
    with torch.no_grad(),runtime.capture() as native:
        baseline=runtime.forward(ids,pos).detach().cpu()
    with torch.no_grad(),runtime.capture(zero_position=pos) as edited:
        zeroed=runtime.forward(ids,pos).detach().cpu()
    counterfactuals=[]
    for unit in controls:
        with torch.no_grad(),runtime.capture(zero_position=pos,neuron_index=unit):
            counterfactuals.append(runtime.forward(ids,pos).detach().cpu())
    lp=baseline.double().log_softmax(-1);lq=zeroed.double().log_softmax(-1)
    p,q=lp.exp(),lq.exp();delta=q-p
    def kl(logits):return float((p*(lp-logits.double().log_softmax(-1))).sum())
    def top(probabilities):
        return [{'token_id':int(i),'token':runtime.tokenizer.decode([int(i)]),'p':float(probabilities[i])}
                for i in probabilities.topk(min(6,len(probabilities))).indices]
    next_id=item.get('following_token_id')
    row={'case_id':item.get('case_id') or item['id'],'condition_id':item['id'],
         'kind':item['kind'],'peak_index':pos,'peak_token':runtime.tokenizer.decode([ids[pos]]),
         'text':runtime.tokenizer.decode(ids,skip_special_tokens=False),
         'saved_product':item['source_product'],'live_product':float(native['product'][pos]),
         'saved_anchor':None,'control_neurons':controls,
         'kl':kl(zeroed),'kl_controls':[kl(x) for x in counterfactuals],
         'top_base':top(p),'top_ablated':top(q),
         'next_token_id':next_id,'next_token':runtime.tokenizer.decode([next_id]) if next_id is not None else None,
         'next_logprob_base':float(lp[next_id]) if next_id is not None else None,
         'next_logprob_ablated':float(lq[next_id]) if next_id is not None else None,
         'gains':[{'token':runtime.tokenizer.decode([int(i)]),'dp':float(delta[i])} for i in delta.topk(5).indices],
         'losses':[{'token':runtime.tokenizer.decode([int(i)]),'dp':float(delta[i])} for i in (-delta).topk(5).indices],
         'edit':edited['edit']}
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path/'ablation-logits.npz',baseline=baseline.numpy(),zeroed=zeroed.numpy(),
                        controls=np.stack([x.numpy() for x in counterfactuals]))
    row['logit_arrays']=str(path/'ablation-logits.npz')
    attribution=item.get('reuse_attribution')
    write(path/'native-ablation.json',{'ablation':row,'native':{k:float(native[k][pos]) for k in ['gate','value','product']},'trace':record_trace(runtime,ids,[None]*len(ids),native)})
    if attribution is None:
        upstream,downstream=direct(runtime,ids,pos,runtime.neuron['layer'],runtime.neuron['index'],
                                   units,cfg['atlas']['top_units'])
    else:upstream,downstream=attribution['upstream'],attribution['downstream']
    return {'status':'complete','native':{k:float(native[k][pos]) for k in ['gate','value','product']},
            'trace':record_trace(runtime,ids,[None]*len(ids),native),
            'ablation':row,'upstream':upstream,'downstream':downstream,
            'readers_examined':units,'attribution_reused':attribution is not None,
            'method':LOCAL_METHOD}

def ablation_section(records):
    rows=[r['ablation'] for r in records if r.get('status')=='complete']
    controls=sorted({n for r in rows for n in r['control_neurons']})
    return {'rows':rows,'replaces_case_ids':[str(r.get('case_id') or r.get('id') or r.get('ablation',{}).get('case_id') or r.get('ablation',{}).get('condition_id')) for r in records],
            'summary':{'n':len(rows),'expected':len(records),'controls':controls,
        'median_kl':float(np.median([r['kl'] for r in rows])) if rows else None,
        'median_kl_controls':float(np.median([k for r in rows for k in r['kl_controls']])) if rows else None}}

def run(request, output, runtime=None):
    output=Path(output);cfg=request_settings(request);cache={}
    request={**request,'model_identity':getattr(runtime,'identity',None),
             'versions':getattr(runtime,'versions',None)}
    for source in [*request.get('prior_registries',[]),str(output/'registry.json')]:
        source=Path(source)
        if source.exists():
            saved=read(source)['conditions']
        else:
            saved=[read(p) for p in (source.parent/'conditions').glob('*/measurement.json')]
        for row in saved:
            if row.get('status')=='complete' or row['id'] not in cache:cache[row['id']]=row
    items=collect(request['sources'],request['report'],cfg)
    width=runtime.model.config.intermediate_size
    candidates=[n for n in range(width) if n!=runtime.neuron['index']]
    control_seed=np.random.SeedSequence([cfg['seed'],runtime.neuron['layer'],runtime.neuron['index']])
    controls=np.random.default_rng(control_seed).choice(candidates,cfg['followups']['random_controls'],replace=False).tolist()
    units=readers(runtime,cfg['atlas']['reader_count'])
    pending={}
    for original in items:
        item=dict(original);key=identity(item,request,cfg);item['id']=key
        source=item.pop('source')
        if key not in pending:
            pending[key]=item;pending[key]['sources']=[source]
        else:
            pending[key]['sources'].append(source)
            if item['split'] in ['train','test']:pending[key]['split']=item['split']
            if item.get('reuse_attribution') is not None:pending[key]['reuse_attribution']=item['reuse_attribution']
    n_new=0
    for key,item in pending.items():
        if key in cache and cache[key].get('status')=='complete':
            aliases=cache[key].setdefault('sources',[])
            for source in item['sources']:
                if source not in aliases:aliases.append(source)
            if item['split'] in ['train','test']:cache[key]['split']=item['split']
            continue
        row={k:v for k,v in item.items() if k!='reuse_attribution'}
        row['measured_at']=now()
        try:
            row.update(measure(runtime,item,units,controls,output/'conditions'/key,cfg))
        except Exception as exc:
            row.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        cache[key]=row;n_new+=1
        dest=output/'conditions'/key;dest.mkdir(parents=True,exist_ok=True)
        write(dest/'measurement.json',row)
        write(output/'progress.json',{'new_conditions':n_new,'known_conditions':len(cache),'latest':key})
    attach_graph_ablations(request['sources'],cache,runtime)
    records=list(cache.values());complete=[r for r in records if r.get('status')=='complete']
    atlas=[]
    for r in complete:
        atlas.append({'id':r['id'],'split':r['split'],'selected':r['trace'][r['position']],
            'upstream':r['upstream'],'downstream':r['downstream'],'trace':r['trace'],
            'text':runtime.tokenizer.decode(r['token_ids'],skip_special_tokens=False),'source_start':0})
    natural=[r for r in atlas if r['split'] in ['train','test']]
    sections={'ablation':ablation_section(records)}
    if natural:
        factors=factorize(atlas,cfg['atlas']['routes_per_pole'],cfg['seed'],cfg['atlas']['nmf_steps'])
        write(output/'route-factors.json',factors)
        sections.update(routes=factors['routes'],routes_matrix=factors['routes_matrix'],
            routes_meta={'n_graphs':len(natural),'n_probe_assignments':len(atlas)-len(natural),'method':factors['method']},
            circuit={'upstream':aggregate(natural,'upstream'),'downstream':aggregate(natural,'downstream'),
                     'n_atlas_graphs':len(natural),'logits':None,'method':LOCAL_METHOD})
    write(output/'atlas-raw.json',atlas)
    write(output/'registry.json',{'settings':cfg,'conditions':records})
    write(output/'sections.json',sections)
    result={'tool':'followup_measurements','status':'complete' if len(complete)==len(records) else 'partial',
            'sections':list(sections),'conditions':len(records),'new_conditions':n_new,
            'reused_conditions':len(records)-n_new,'completed_conditions':len(complete),
            'random_controls':controls,'registry':str(output/'registry.json'),
            'route_prediction':None,'telemetry':runtime.telemetry()}
    write(output/'result.json',result)
    return result


def attach_graph_ablations(sources, cache, runtime):
    # Only the graph files produced by this group are completed here.
    for entry in sources:
        if entry['tool']!='attribute' or not entry.get('current'):continue
        folder=Path(entry['output'])
        match=next((r for r in cache.values() if r.get('status')=='complete'
                    and any(x.get('folder')==str(folder) for x in r['sources'])),None)
        if match is None:continue
        saved=read(folder/'result.json');req=read(folder/'tool-request.json')
        arrays=np.load(match['ablation']['logit_arrays'])
        a=runtime.torch.from_numpy(arrays['baseline']);b=runtime.torch.from_numpy(arrays['zeroed']);arrays.close()
        reference=req['root'].get('token_id')
        if reference is None:reference=int(a.argmax())
        pa,pb=a.double().softmax(-1),b.double().softmax(-1)
        measured=match['ablation']
        saved['ablation']={'status':'complete','position':match['position'],'edit':measured['edit'],
            'baseline_target_activation':match['native']['product'],
            'reference_token_id':reference,'reference_token':runtime.tokenizer.decode([reference]),
            'reference_is_requested_logit':req['root']['kind']=='output_logit',
            'delta_target_logit':float(b[reference]-a[reference]),
            'delta_target_prob':float(pb[reference]-pa[reference]),'kl':measured['kl'],
            'baseline_top':measured['top_base'],'zeroed_top':measured['top_ablated'],
            'control_neurons':measured['control_neurons'],'kl_controls':measured['kl_controls'],
            'registry_condition':match['id'],'logit_arrays':measured['logit_arrays']}
        saved['graph']['meta']['ablation']=saved['ablation']
        write(folder/'result.json',saved)
        graph=read(folder/'graph.json');graph['graph']=saved['graph'];write(folder/'graph.json',graph)


def combine_ablations(previous, incoming):
    rows={}
    replacing=set((incoming or {}).get('replaces_case_ids',[]))
    for packet in [previous or {},incoming or {}]:
        for original in packet.get('rows',[]):
            if packet is previous and str(original.get('case_id') or original.get('condition_id')) in replacing:continue
            row=copy.deepcopy(original)
            row.setdefault('control_neurons',packet.get('summary',{}).get('controls',[]))
            key=str(row.get('case_id') or row.get('condition_id'))
            rows[key]=row
    result=ablation_section([{'status':'complete','ablation':row} for row in rows.values()])
    planned=set(result['replaces_case_ids']) | set((incoming or {}).get('replaces_case_ids',[]))
    result['replaces_case_ids']=sorted(planned)
    result['summary']['expected']=len(planned)
    return result
