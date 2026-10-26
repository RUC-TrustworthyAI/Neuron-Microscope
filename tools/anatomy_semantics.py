"""Label measured probe anchors; reuse saved assignments across report renders."""
import collections
import copy
import hashlib
import html
import json
from pathlib import Path
from common import read, write, now

SYSTEM = """Classify measured contexts into semantic groups for a neuron report.
The neuron activation is product; gate and value are supporting measurements. Read the observed prefix through the measured anchor. Following text is a prediction target, not information available at the anchor.
Reuse the supplied semantic groups when they fit. Keep their IDs, meanings and response poles. Add a group when a recurring measured context has a different meaning; describe it from the supplied evidence. For a new group, set pole to positive, negative or either. A response pole describes the group's semantic association with the neuron, not sentiment or a vote over member signs. Use either for mixed, background or unresolved evidence.
Assign every requested point. A measurement's sign, search title or intended control role alone does not establish its semantic group.
Organize the semantic groups into clusters for the gate/value scatter. Group related meanings; put the main meaning first in member_groups, then its related variants. Each semantic group belongs to one cluster. Preserve supplied prior clusters where they still fit. These are semantic clusters, not inferred causal routes. Do not choose group membership or order to obtain a particular colour.
Return JSON:
{"new_groups":[{"id":"new ID","name":"short name","description":"meaning","pole":"positive|negative|either","reason":"evidence"}],
 "labels":[{"id":"requested point ID","group":"semantic group ID","reason":"short explanation"}],
 "clusters":[{"id":"cluster ID","name":"short name","member_groups":["main semantic group ID","related group ID"],"reason":"semantic relationship"}]}"""


def probe_key(point):
    value=([point['dataset'],point['neuron'],point['dataset_token_index'],
            point['gate'],point['up'],point['activation']]
           if point.get('source')=='corpus:scan' else
           [point['token_ids'],point['position'],point['gate'],point['up'],point['activation']])
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def prior_labels(request):
    result={}
    for directory in request.get('sections',[]):
        path=Path(directory)/'anatomy-semantics.json'
        if path.exists():result=read(path)
    if not result and request.get('report'):
        paths=list((Path(request['report'])/'artifacts').glob('*/anatomy-semantics.json'))
        if paths:result=max((read(path) for path in paths),key=lambda x:x.get('created_at',''))
    return result


def checkpoint_labels(folder):
    """Reuse completed annotation batches after a process or host interruption."""
    folder=Path(folder)
    if (folder/'anatomy-semantics.json').exists():
        return read(folder/'anatomy-semantics.json')
    checkpoint=folder/'assignments-checkpoint.json'
    if not checkpoint.exists():return {}
    saved=read(folder/'evidence.json') if (folder/'evidence.json').exists() else {}
    definitions={g['id']:{k:v for k,v in g.items() if k!='saved_examples'}
                 for g in saved.get('semantic_groups',[])}
    clusters=saved.get('prior_clusters',[])
    outputs=sorted(folder.glob('batch-*/call/output.json'))
    outputs+=sorted(folder.glob('resume-*/**/call/output.json'))
    if (folder/'call/output.json').exists():outputs.insert(0,folder/'call/output.json')
    for path in outputs:
        answer=read(path)
        for group in answer.get('new_groups',[]):
            if group.get('id') and group['id'] not in definitions and group.get('pole') in ('positive','negative','either'):
                definitions[group['id']]=group
        clusters=answer.get('clusters',clusters)
    return {'assignments':read(checkpoint),'groups':list(definitions.values()),
            'clusters':clusters,'labeler':'DeepSeek high'}


def packet(bundle, pending, previous):
    anatomy=bundle['anatomy'];spectrum=bundle['spectrum']
    definitions={g['id']:g for g in spectrum.get('groups',[])}
    definitions.update({g['id']:g for g in previous.get('groups',[]) if g['id'] not in definitions})
    examples=collections.defaultdict(list)
    for point in anatomy['points']:
        if point.get('source_family')!='corpus':continue
        group=point.get('semantic_group',point['cluster'])
        examples[group].append(point)
    groups=[]
    for group in definitions.values():
        members=sorted(examples[group['id']],key=lambda p:-abs(p['activation']))[:3]
        groups.append({k:v for k,v in group.items() if k not in ['case_ids','n']})
        groups[-1]['saved_examples']=[{'product':p['activation'],'observed_prefix':html.unescape(p['text'])} for p in members]
    points=[]
    for point in pending:
        prefix=point.get('observed_prefix',html.unescape(point['text']));full=point.get('input_text','')
        points.append({'id':point['id'],'observed_prefix':prefix,
            'following_prediction_target':full[len(prefix):] if full.startswith(prefix) else '',
            'anchor':{'position':point['position'],'token':html.unescape(point['peak_token']),
                      'gate':point['gate'],'value':point['up'],'product':point['activation']}})
    return {'semantic_groups':groups,'prior_clusters':previous.get('clusters',[]),
            'experimental_points':points}


def annotate(bundle, folder, previous=None, llm=None, batch_size=32):
    if llm is None:
        from blind_tests import call
        llm=call
    previous=previous or {};folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    probes=[p for p in bundle['anatomy']['points']
            if p.get('source_family')=='probe' or p.get('source')=='corpus:scan']
    assignments=copy.deepcopy(previous.get('assignments',{}))
    for p in probes:
        if p.get('source')=='corpus:scan' and p.get('semantic_group'):
            assignments.setdefault(probe_key(p),{'id':p['id'],'group':p['semantic_group'],
                                                  'reason':'Reused saved natural-corpus assignment.'})
    pending=[p for p in probes if probe_key(p) not in assignments]
    evidence=packet(bundle,pending,previous);write(folder/'evidence.json',evidence)
    definitions={g['id']:{k:v for k,v in g.items() if k!='saved_examples'} for g in evidence['semantic_groups']}
    clusters=copy.deepcopy(previous.get('clusters',[]));calls=0
    batches=([pending[i:i+batch_size] for i in range(0,len(pending),batch_size)]
             if any(p.get('source')=='corpus:scan' for p in pending) else [pending])
    if not pending and clusters:batches=[]
    call_records=read(folder/'calls.json') if (folder/'calls.json').exists() else []
    if not call_records:
        for path in sorted(folder.glob('batch-*/messages.json')):
            if (path.parent/'call/output.json').exists():
                context=__import__('json').loads(read(path)[1]['content'])
                call_records.append({'directory':str(path.parent),'requested':len(context['experimental_points'])})
    call_root=folder
    if pending and (list(folder.glob('batch-*')) or (folder/'call').exists()):
        number=1+len(list(folder.glob('resume-*')))
        call_root=folder/('resume-%03d'%number)
    for index,batch in enumerate(batches,1):
        context=packet(bundle,batch,{'groups':list(definitions.values()),'clusters':clusters})
        messages=[{'role':'system','content':SYSTEM},
                  {'role':'user','content':json.dumps(context,ensure_ascii=False)}]
        call_folder=call_root if len(batches)==1 else call_root/('batch-%03d'%index)
        call_folder.mkdir(parents=True,exist_ok=True)
        write(call_folder/'messages.json',messages)
        answer=llm(messages,call_folder/'call','anatomy-semantics',max_tokens=393216);calls+=1
        for group in answer.get('new_groups',[]):
            if group.get('id') and group['id'] not in definitions and group.get('pole') in ['positive','negative','either']:
                definitions[group['id']]=group
        by_id={p['id']:p for p in batch}
        for row in answer.get('labels',[]):
            if row.get('id') in by_id and row.get('group') in definitions:
                assignments[probe_key(by_id[row['id']])]=row
        clusters=answer.get('clusters',clusters)
        call_records.append({'directory':str(call_folder),'requested':len(batch)})
        write(folder/'label-progress.json',{'completed_calls':calls,'total_calls':len(batches),
              'assigned':sum(probe_key(p) in assignments for p in probes),'requested':len(probes),
              'finished_at':now()})
        write(folder/'assignments-checkpoint.json',assignments)
    # Unassigned groups remain visible; there is no sign-based classification fallback.
    used=set();normal=[]
    for cluster in clusters:
        members=[g for g in cluster.get('member_groups',[]) if g in definitions and g not in used]
        if not members:continue
        used.update(members);normal.append({**cluster,'member_groups':members})
    for identity,group in definitions.items():
        if identity not in used:
            normal.append({'id':'group-'+identity,'name':group['name'],'member_groups':[identity]})
    result={'groups':list(definitions.values()),'clusters':normal,'assignments':assignments,
            'labeler':'DeepSeek high','created_at':now()}
    write(folder/'anatomy-semantics.json',result)
    unresolved=[p['id'] for p in probes if probe_key(p) not in assignments]
    write(folder/'calls.json',call_records)
    write(folder/'sections.json',{})
    write(folder/'tool-request.json',{'id':folder.name,'output':str(folder),'only':'anatomy_semantics'})
    write(folder/'result.json',{'status':'partial' if unresolved else 'complete','llm_requests':calls,
        'n_probes':sum(p.get('source_family')=='probe' for p in probes),
        'n_scan_points':sum(p.get('source')=='corpus:scan' for p in probes),
        'n_retained':len(probes)-len(pending),'n_requested':len(pending),
        'n_assigned':len(probes)-len(unresolved),'unresolved':unresolved,'new_model_forwards':0})
    return result


def apply(bundle, semantics):
    anatomy=bundle.get('anatomy')
    if not anatomy or not semantics:return
    definitions={g['id']:g for g in semantics['groups']}
    memberships={group:c['id'] for c in semantics['clusters'] for group in c['member_groups']}
    assignments=semantics['assignments'];quadrants=collections.defaultdict(collections.Counter)
    for point in anatomy['points']:
        if point.get('source_family')=='probe' or point.get('source')=='corpus:scan':
            label=assignments.get(probe_key(point),{})
            group=label.get('group',point.get('semantic_group',point['cluster']))
            if label:point['label_source']='saved-scan-label' if point.get('source')=='corpus:scan' else 'saved-probe-label';point['label_reason']=label.get('reason','')
        else:group=point.get('semantic_group',point['cluster'])
        point['semantic_group']=group
        point['cluster']=memberships.get(group,group)
        q=('g+' if point['gate']>0 else 'g-')+('u+' if point['up']>0 else 'u-')
        quadrants[point['cluster']][q]+=1
    anatomy['semantic_groups']=semantics['groups']
    anatomy['clusters']=semantics['clusters']
    anatomy['group_names']={c['id']:c['name'] for c in semantics['clusters']}
    anatomy['group_names']['unlabelled-probes']='Unlabelled experimental contexts'
    anatomy['quadrants']={k:dict(v) for k,v in quadrants.items()}
    anatomy['semantic_labeler']=semantics['labeler']
    anatomy['selection']['labels']='Saved natural assignments and DeepSeek annotations of measured scan and experimental prefixes.'


def label_request(request, folder, llm=None, *, include_scan=True):
    """Label new measured contexts; normal stage two reuses the prepared scan."""
    from report import source_bundle
    from anatomy_selection import build
    report=Path(request['report']);bundle,_=source_bundle((report/'index.html').read_text())
    data=read(report/'report.json')
    for directory in request.get('sections',[]):
        for key,value in read(Path(directory)/'sections.json').items():bundle[key]=copy.deepcopy(value)
    annotations=read(request['annotations']) if request.get('annotations') else {}
    if annotations.get('anatomy'):
        from paper_sections import prepare
        bundle['anatomy']=prepare({'anatomy':annotations['anatomy']},report)['_anatomy']
    if not (bundle.get('spectrum') or {}).get('points'):return None
    build(bundle,data,request)
    from anatomy_scan import ensure
    scan_folder=ensure(request,bundle,folder) if include_scan else None
    if scan_folder:build(bundle,data,request)
    previous=prior_labels(request)
    probes=[p for p in bundle['anatomy']['points']
            if p.get('source_family')=='probe' or p.get('source')=='corpus:scan']
    if not scan_folder and previous.get('clusters') and all(probe_key(p) in previous.get('assignments',{}) for p in probes):return None
    annotate(bundle,folder,previous,llm,
             batch_size=request.get('settings',{}).get('anatomy_scan',{}).get('label_batch_size',32))
    return Path(folder)
