"""One measured packet for search, anatomy, routes, family and local circuit."""
import collections
import copy
import hashlib
import math
from pathlib import Path
import numpy as np
from common import read, write, tokenize, position, score_positions, select_peak
from settings import load, request_settings
from model_tools import record_trace

LOCAL_METHOD = "Local RelP: source product times the modified reader derivative, with intervening MLP outputs detached. Upstream values pool the prefix; downstream values cover nominated same-position readers."

def coord(u):
    return int(u['layer']), int(u.get('neuron', u.get('index')))

def exemplar(row):
    t = row['trace']; p = row['selected']['position']
    return {'text': row['text'], 'activation': row['selected']['product'],
            'token_activations': [{'token':x['token'],'activation':x['product']} for x in t],
            'peak_index':p, 'cut_left':bool(row.get('source_start',0)), 'cut_right':False}

def prepare(plan, report):
    """Resolve saved case references on the host; preserve their actual input token IDs."""
    data = read(Path(report)/'report.json')
    plan = copy.deepcopy(plan)
    def case(item, window=None):
        if isinstance(item,(str,int)):item={'case_id':str(item)}
        item=copy.deepcopy(item)
        if 'case_id' not in item:return item
        point=data['points'][str(item['case_id'])];doc=data['documents'][point['document']]
        end=point['position']+1
        item.update(token_ids=doc['token_ids'][:end],position=end-1,
                    source_start=0,source_document=point['document'],
                    source_position=point['position'],original_product=point['values'][2],
                    following_token_id=doc['token_ids'][end] if end<len(doc['token_ids']) else None,
                    score={'location':'token_position','position':end-1,'reduction':'single'})
        return item
    atlas=plan.get('atlas')
    if atlas:
        atlas['cases']=[case(x) for x in atlas.get('cases',[])]
    family=plan.get('family')
    if family:
        family['cases']=[case(x) for x in family.get('cases',[])]
    anatomy=plan.get('anatomy')
    if anatomy:
        memberships={str(p):g['id'] for g in anatomy.get('groups',[]) for p in g.get('case_ids',[])}
        points=[]
        selected=set(map(str,data.get('scatter',[])))|set(memberships)
        for p in sorted(selected,key=int):
            a=data['points'][p];d=data['documents'][a['document']];j=a['position']
            points.append({'id':p,'gate':a['values'][0],'up':a['values'][1],
                           'activation':a['values'][2],'peak_token':d['tokens'][j],
                           'text':''.join(d['tokens'][max(0,j-24):j+1]),
                           'cluster':memberships.get(p,'unassigned')})
        quadrants={}
        for p in points:
            key=('g+' if p['gate']>0 else 'g-' if p['gate']<0 else 'g0')+('u+' if p['up']>0 else 'u-' if p['up']<0 else 'u0')
            bucket=quadrants.setdefault(p['cluster'],{});bucket[key]=bucket.get(key,0)+1
        packet={'points':points,'quadrants':quadrants,'n':len(points),
                'group_names':{g['id']:g.get('name',g['id']) for g in anatomy.get('groups',[])}}
        def strip(selection):
            if isinstance(selection,(str,int)):selection={'case_id':str(selection)}
            a=data['points'][str(selection['case_id'])];d=data['documents'][a['document']]
            j=a['position'];start=max(0,j-int(selection.get('before',24)))
            end=min(len(d['tokens']),j+int(selection.get('after',10))+1)
            return {'text':''.join(d['tokens'][start:end]),'tokens':d['tokens'][start:end],
                    'gate':d['gate'][start:end],'up':d['value'][start:end],'act':d['product'][start:end],
                    'case_id':str(selection['case_id']),'source_start':start,'anchor_index':j-start,
                    'caption':selection.get('caption','')}
        if anatomy.get('sentence'):packet['strip']=strip(anatomy['sentence'])
        packet['headline']=[strip(x) for x in anatomy.get('pairs',[])]
        plan['_anatomy']=packet
    plan['_corpus']={'max':data['statistics']['product']['max'],
                     'min':data['statistics']['product']['min'],
                     'n_tokens':data['statistics']['product']['count'] if 'count' in data['statistics']['product'] else data.get('n_tokens')}
    return plan

def input_case(runtime, item, limit):
    if isinstance(item,str):item={'text':item}
    if 'token_ids' in item:
        ids=item['token_ids']
        text=runtime.tokenizer.decode(ids,skip_special_tokens=False)
        offsets=[None]*len(ids)
    else:
        text=item['text'];ids,offsets=tokenize(runtime.tokenizer,text,limit)
    pos=position(item.get('position',-1),len(ids))
    if item.get('score'):
        selected=score_positions(text,offsets,(0,len(text)),item['score'])
    else:selected=list(range(len(ids)))
    return text,ids,offsets,pos,selected

def probe(runtime,item,limit=None):
    text,ids,offsets,pos,selected=input_case(runtime,item,limit)
    with runtime.torch.no_grad(), runtime.capture() as state:runtime.forward(ids)
    trace=record_trace(runtime,ids,offsets,state)
    return {'text':text,'token_ids':ids,'trace':trace,'scored_positions':selected,
            'selected':trace[pos], **{k:item[k] for k in ['case_id','source_start','source_document','source_position','original_product','following_token_id','semantic_group','score'] if k in item}}

def search(runtime,regions,limit,folder):
    previous=read(folder/'search-raw.json') if (folder/'search-raw.json').exists() else []
    rows=[]
    for ri,region in enumerate(regions):
        rec={**{k:region[k] for k in ['id','name','pole','description','rival'] if k in region},
             'candidates':[],'controls':[]}
        rec.setdefault('id','region-'+str(ri+1));rec.setdefault('pole','positive')
        for key in ['candidates','controls']:
            for i,item in enumerate(region.get(key,[])):
                if isinstance(item,str):item={'text':item}
                prior=previous[ri].get(key,[]) if ri<len(previous) else []
                if i<len(prior) and prior[i].get('status')=='complete':
                    rec[key].append(prior[i]);continue
                try:
                    item=dict(item)
                    item['score']=item.get('score') or region['score']
                    row=probe(runtime,item,limit)
                    row['score']=item['score']
                    row['selected']=select_peak([row['trace'][p] for p in row['scored_positions']],item['score']['reduction'])
                    row['status']='complete'
                except Exception as e:row={'status':'failed','input':item,'error':str(e)}
                rec[key].append(row);write(folder/'search-raw.json',rows+[rec])
        rows.append(rec)
    return rows

def readers(runtime,n=None):
    """Cheap geometry nominates readers; atlas derivatives measure their contributions."""
    if n is None:n=load()['atlas']['reader_count']
    torch=runtime.torch;layer,unit=coord(runtime.neuron);candidates=[]
    with torch.no_grad():
        v=runtime.model.model.layers[layer].mlp.down_proj.weight[:,unit]
        for l in range(layer+1,len(runtime.model.model.layers)):
            mlp=runtime.model.model.layers[l].mlp;d=v.to(mlp.up_proj.weight.device)
            scores=(mlp.up_proj.weight@d).float().abs()+(mlp.gate_proj.weight@d).float().abs()
            for score,index in zip(*torch.topk(scores,min(n,len(scores)))):
                candidates.append({'layer':l,'neuron':int(index),'selection_score':float(score)})
    return sorted(candidates,key=lambda x:-x['selection_score'])[:n]

def local_graph(runtime,item,reader_units,limit=None,top_units=None):
    if top_units is None:top_units=load()['atlas']['top_units']
    from relp import direct
    text,ids,offsets,pos,_=input_case(runtime,item,limit)
    layer,unit=coord(runtime.neuron)
    with runtime.torch.no_grad(),runtime.capture() as state:
        runtime.forward(ids)
    cache=getattr(runtime,'_local_graph_cache',None)
    if cache is None:cache=runtime._local_graph_cache={}
    key=(tuple(ids),pos,layer,unit,tuple(coord(u) for u in reader_units),top_units)
    if key in cache:upstream,downstream=cache[key]
    else:
        upstream,downstream=direct(runtime,ids,pos,layer,unit,reader_units,top_units)
        cache[key]=(upstream,downstream)
    trace=record_trace(runtime,ids,offsets,state)
    return {'text':text,'token_ids':ids,'trace':trace,'selected':trace[pos],
        'upstream':upstream,'downstream':downstream,'method':LOCAL_METHOD,
        'source_positions':[0,pos],'retained_upstream_units':top_units,
        'readers_examined':reader_units,'downstream_coverage':'nominated readers, same token position',
        'observed_prefix':runtime.tokenizer.decode(ids[:pos+1],skip_special_tokens=False),
        **{k:item[k] for k in ['case_id','source_start','source_document','source_position','original_product','split','following_token_id','semantic_group','score'] if k in item}}

def aggregate(rows,key):
    acc={}
    for row in rows:
        for u in row.get(key,[]):
            k=coord(u);a=acc.setdefault(k,{'layer':k[0],'neuron':k[1],'weight':0.,'count':0,'abs_sum':0.})
            a['weight']+=u['weight'];a['abs_sum']+=abs(u['weight']);a['count']+=int(u['weight']!=0)
    return sorted(acc.values(),key=lambda x:-x['abs_sum'])

def factorize(rows,k=None,seed=None,steps=None):
    """Signed upstream vectors -> nonnegative factors, fitted on training documents only."""
    cfg=load();k=cfg['atlas']['routes_per_pole'] if k is None else k
    seed=cfg['seed'] if seed is None else seed;steps=cfg['atlas']['nmf_steps'] if steps is None else steps
    result={'routes':[],'routes_matrix':{},'assignments':{},'method':'NMF of L1-normalized, sign-split local RelP MLP contributions; training documents only.'}
    rng=np.random.default_rng(seed)
    for pole,key,prefix in [(1,'pos','p'),(-1,'neg','n')]:
        selected=[x for x in rows if x['selected']['product']*pole>0]
        train=[x for x in selected if x['split']=='train']
        features=sorted({(u['layer'],u['neuron'],1 if u['weight']>0 else -1)
                         for x in train for u in x['upstream'] if u['weight']})
        if not train or not features:continue
        columns={f:i for i,f in enumerate(features)}
        X=np.zeros((len(selected),len(features)),dtype=np.float64)
        for i,x in enumerate(selected):
            for u in x['upstream']:
                f=(u['layer'],u['neuron'],1 if u['weight']>0 else -1)
                if f in columns:X[i,columns[f]]+=abs(u['weight'])
        norms=X.sum(1,keepdims=True);X/=np.maximum(norms,1e-12)
        train_ix=[i for i,x in enumerate(selected) if x['split']=='train']
        T=X[train_ix];rank=min(k,len(train),len(features))
        W=rng.random((len(T),rank))+.1;H=rng.random((rank,len(features)))+.1
        for _ in range(steps):
            H*= (W.T@T)/(W.T@W@H+1e-12)
            W*= (T@H.T)/(W@H@H.T+1e-12)
        # Fix factor scale before comparing route weights.
        scale=np.maximum(H.sum(1),1e-12);H/=scale[:,None]
        weights=np.full((len(X),rank),1./rank)
        for _ in range(steps):weights*= (X@H.T)/(weights@H@H.T+1e-12)
        ids=[prefix+str(i+1) for i in range(rank)]
        for row,w in zip(selected,weights):
            result['assignments'][row['id']]=ids[int(w.argmax())] if w.sum()>0 else None
        for j,rid in enumerate(ids):
            members=[x for x in train if result['assignments'][x['id']]==rid]
            units=[]
            for index in np.argsort(-H[j])[:8]:
                a,b,sign=features[index]
                units.append({'layer':a,'neuron':b,'weight':float(H[j,index])*sign})
            result['routes'].append({'id':rid,'pole':pole,'name':'Route '+rid,
                'n_graphs':len(members),'graph_ids':[x['id'] for x in members],
                'share':sum(sum(abs(u['weight']) for u in x['upstream']) for x in members),
                'upstream':units,'examples':[exemplar(x) for x in members[:3]]})
        result['routes_matrix'][key]={'graph_ids':[x['id'] for x in selected],
            'graph_texts':[x['text'] for x in selected],'graph_activations':[x['selected']['product'] for x in selected],
            'route_ids':ids,'route_weights':weights.tolist(),'training_graph_ids':[x['id'] for x in train],
            'features':[list(f) for f in features],'components':H.tolist(),
            'training_reconstruction_error':float(np.linalg.norm(T-weights[train_ix]@H))}
    total=sum(r['share'] for r in result['routes']) or 1
    for r in result['routes']:r['share']/=total
    return result

def joint(runtime,cases,units,limit):
    measurements=[]
    for item in cases:
        text,ids,offsets,pos,_=input_case(runtime,item,limit);observed={};handles=[]
        def capture(layer,coords):
            def f(module,args):
                for _,unit in coords:observed[(layer,unit)]=args[0][0,:,unit].detach().float().cpu().tolist()
            return f
        try:
            layers=collections.defaultdict(list)
            for u in units:layers[coord(u)[0]].append(coord(u))
            for l,coords in layers.items():
                handles.append(runtime.model.model.layers[l].mlp.down_proj.register_forward_pre_hook(capture(l,coords)))
            with runtime.torch.no_grad():runtime.forward(ids)
            measurements.append({'prompt':text,'tokens':[runtime.tokenizer.decode([i]) for i in ids],
                'input_token_ids':ids,'readouts':[{'layer':l,'neuron':n,'values':v} for (l,n),v in observed.items()],
                **{k:item[k] for k in ['case_id','source_start','source_position'] if k in item}})
        finally:
            for h in handles:h.remove()
    return measurements

def run(request,output,runtime=None):
    output=Path(output);plan=request['plan'];raw={};patch={};cfg=request_settings(request)
    if plan.get('_anatomy'):patch['anatomy']=plan['_anatomy']
    write(output/'sections.json',patch)
    if plan.get('search'):
        raw['search']=search(runtime,plan['search'],request['limits']['max_input_tokens'],output)
        patch.update(search_patch(raw['search'],plan.get('_corpus',{})))
        write(output/'sections.json',patch)
    atlas=plan.get('atlas')
    if atlas:
        selected_readers=atlas.get('readers')
        if selected_readers is None:selected_readers=readers(runtime,cfg['atlas']['reader_count'])
        previous=read(output/'atlas-raw.json') if (output/'atlas-raw.json').exists() else []
        raw['atlas']=[]
        for i,item in enumerate(atlas.get('cases',[])):
            if i<len(previous) and previous[i].get('status')=='complete':
                raw['atlas'].append(previous[i]);continue
            try:
                row=local_graph(runtime,item,selected_readers,cfg['inputs']['max_written_tokens'],cfg['atlas']['top_units'])
                doc=str(item.get('source_document') or item.get('text') or item['token_ids'])
                fraction=int.from_bytes(hashlib.sha256((str(request.get('seed',cfg['seed']))+doc).encode()).digest()[:8],'big')/2**64
                row['split']=item.get('split', 'test' if fraction<cfg['atlas']['holdout_fraction'] else 'train')
                row.update(id=request['id']+'-a'+str(i+1),status='complete')
            except Exception as e:row={'id':request['id']+'-a'+str(i+1),'status':'failed','input':item,'error':str(e)}
            raw['atlas'].append(row);write(output/'atlas-raw.json',raw['atlas'])
        complete=[x for x in raw['atlas'] if x['status']=='complete']
        factors=factorize(complete,cfg['atlas']['routes_per_pole'],request.get('seed',cfg['seed']),cfg['atlas']['nmf_steps'])
        write(output/'route-factors.json',factors)
        patch.update(routes=factors['routes'],routes_matrix=factors['routes_matrix'],
            routes_meta={'n_graphs':len(complete),'method':factors['method']},
            circuit={'upstream':aggregate(complete,'upstream'),'downstream':aggregate(complete,'downstream'),
                     'n_atlas_graphs':len(complete),'logits':None,'method':LOCAL_METHOD})
        raw['route_assignments']=factors['assignments']
        write(output/'sections.json',patch)
    family=plan.get('family')
    if family:
        units={coord(u):u for u in family.get('units',[])}
        circuit=patch.get('circuit',{})
        for side in ['upstream','downstream']:
            for u in circuit.get(side,[])[:cfg['family']['neighbours_per_side']]:
                units.setdefault(coord(u),{'layer':u['layer'],'neuron':u['neuron']})
        units.setdefault(coord(request['neuron']),{'layer':request['neuron']['layer'],'neuron':request['neuron']['index']})
        family_key={'cases':family.get('cases',[]),'units':list(units.values())}
        saved=read(output/'family-raw.json') if (output/'family-raw.json').exists() else {}
        if saved.get('input')==family_key:measurements=saved['measurements']
        else:
            measurements=joint(runtime,family.get('cases',[]),list(units.values()),cfg['inputs']['max_written_tokens'])
            write(output/'family-raw.json',{'input':family_key,'measurements':measurements})
        patch['family']={'family':list(units.values()),'texts':[m['prompt'] for m in measurements],
                         'measurements':measurements,**{k:circuit.get(k,[]) for k in ['upstream','downstream']}}
    patch.update(search_patch(raw.get('search',[]),plan.get('_corpus',{})))
    write(output/'sections.json',patch);write(output/'raw.json',raw)
    result={'tool':'paper_sections','status':'complete','sections':list(patch),
            'atlas_completed':sum(x.get('status')=='complete' for x in raw.get('atlas',[])),
            'atlas_expected':len(raw.get('atlas',[])), 'telemetry':runtime.telemetry() if runtime else {'forwards':0},
            'model':getattr(runtime,'identity',None)}
    failures=sum(x.get('status')!='complete' for x in raw.get('atlas',[]))
    failures+=sum(x.get('status')!='complete' for r in raw.get('search',[]) for k in ['candidates','controls'] for x in r[k])
    if failures:result.update(status='incomplete',failed_measurements=failures)
    return result

def search_patch(rows,corpus):
    if not rows:return {}
    regions=[];all_pos=[];all_neg=[];count=0
    for r in rows:
        candidates=[x for x in r['candidates'] if x['status']=='complete']
        controls=[x for x in r['controls'] if x['status']=='complete']
        count+=len(candidates)+len(controls)
        vals=[x['selected']['product'] for x in candidates]
        neg=r.get('pole')=='negative'
        ranked=sorted(candidates,key=lambda x:x['selected']['product'],reverse=not neg)
        (all_neg if neg else all_pos).extend(vals)
        regions.append({k:v for k,v in r.items() if k not in ['candidates','controls']}|{'n':len(candidates),
            'best':(min(vals) if neg else max(vals)) if vals else None,
            'frontier':[exemplar(x) for x in ranked[:3]],
            'control':{'n':len(controls),'candidate_median':float(np.median(vals)) if vals else None,
                       'control_median':float(np.median([x['selected']['product'] for x in controls])) if controls else None}})
    return {'search':{'regions':regions,'regions_more':[],'n_regions':len(regions),'n_texts':count,
        'corpus_max':corpus.get('max'),'corpus_min':corpus.get('min'),
        'best_positive':max(all_pos) if all_pos else None,'best_negative':min(all_neg) if all_neg else None}}

def labelled(patch,labels):
    """Labels change prose and membership; every numeric field stays measured."""
    patch=copy.deepcopy(patch)
    route_labels=labels.get('routes',{})
    for r in patch.get('routes',[]):
        r.update({k:v for k,v in route_labels.get(r['id'],{}).items() if k in ['name','description']})
    unit_labels=labels.get('units',{})
    for section,keys in [('family',['family','upstream','downstream']),('circuit',['upstream','downstream'])]:
        for key in keys:
            for u in patch.get(section,{}).get(key,[]):
                ident='L%d_N%d'%coord(u)
                if ident in unit_labels:u['label']=unit_labels[ident]
    return patch
