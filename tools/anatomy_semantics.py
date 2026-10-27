"""Label measured probe anchors; reuse saved assignments across report renders."""
import collections
import copy
import hashlib
import html
import json
from pathlib import Path
from common import read, write, now
from measured_response import response_pole, summary, describe_group, semantic_definition, mixed_batches

SYSTEM = """Classify measured contexts into semantic groups for a neuron report.
The neuron activation is product. response_pole is computed from its sign: positive for product > 0, negative for product < 0, zero for product = 0. These are observed facts at the specified token, including when raw gate is negative. Gate and value are supporting measurements. Do not assign or change response poles for points or groups.
Read the observed prefix through the measured anchor. Following text is a prediction target, not information available at the anchor. Explain recurring conditions in the measured positive and negative responses separately. A small nonzero response has a sign but does not by itself support a strong semantic association. Compare magnitudes relative to the supplied observed range.
Maintain one compact semantic taxonomy for the whole neuron across every batch, typically ten to fourteen groups including broad background and unresolved contexts. Use specific contextual distinctions supported by multiple samples; a new entity, article topic or isolated example does not require another group. Names and definitions must describe their members. Shared subject matter alone does not establish why their responses have opposite signs. Preserve all examples and actual measurements.
Review each group's measured positive and negative members and response_summary. Revise a broad or contradicted definition through group_updates; existing definitions are provisional. Return revised labels for previously seen IDs in group saved_examples or labelled_points when needed. Repeated strong responses inconsistent with a directional explanation require revising that explanation or distinguishing supported contextual conditions; do not automatically describe every disagreement as an exception. Leave unresolved mechanisms explicit.
Reuse compatible groups and merge redundant definitions with group_merges. The retained group keeps its ID; preserve evidence through the mapping. Group descriptions define meanings, not a predicted activation sign. Return no pole field. Organize related semantic groups into roughly six to eight readable scatter clusters using member_groups. Mixed responses may share a semantic category, and their measured signs remain visible individually. Do not choose membership or ordering to obtain a colour.
Assign every requested point. Sign and magnitude constrain the response explanation but do not by themselves identify the text's meaning. Use the complete group inventory, including previous assignments, when revising the taxonomy.
Return JSON:
{"new_groups":[{"id":"new ID","name":"short name","description":"meaning and supported response conditions","reason":"evidence"}],
 "group_updates":[{"id":"existing ID","name":"revised short name","description":"revised definition","reason":"evidence requiring revision"}],
 "group_merges":[{"source":"existing group ID","target":"retained group ID","reason":"compatible meanings"}],
 "labels":[{"id":"requested or previously labelled point ID","group":"semantic group ID","reason":"contextual membership"}],
 "clusters":[{"id":"cluster ID","name":"short name","member_groups":["semantic group ID"],"reason":"semantic relationship"}]}"""


def canonical(group, aliases):
    seen=set()
    while group in aliases and group not in seen:
        seen.add(group);group=aliases[group]
    return group


def merge_groups(definitions, assignments, clusters, aliases, merges, core=()):
    """Apply explicit model-authored merges; retain all samples and raw call records."""
    for row in merges:
        source=row.get('source');target=canonical(row.get('target'),aliases)
        if source not in definitions or target not in definitions or source==target:
            continue
        aliases[source]=target;definitions.pop(source)
    for source in list(aliases):aliases[source]=canonical(aliases[source],aliases)
    for row in assignments.values():row['group']=canonical(row['group'],aliases)
    for cluster in clusters:
        cluster['member_groups']=list(dict.fromkeys(canonical(g,aliases) for g in cluster.get('member_groups',[])))


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
    definitions={g['id']:semantic_definition(g)
                 for g in saved.get('semantic_groups',[])}
    clusters=saved.get('prior_clusters',[]);aliases=dict(saved.get('aliases',{}))
    core=set(saved.get('core_group_ids',[g['id'] for g in saved.get('semantic_groups',[])]))
    assignments=read(checkpoint)
    outputs=sorted(folder.glob('batch-*/call/output.json'))
    outputs+=sorted(folder.glob('resume-*/**/call/output.json'))
    if (folder/'call/output.json').exists():outputs.insert(0,folder/'call/output.json')
    for path in outputs:
        answer=read(path)
        for group in answer.get('new_groups',[]) + answer.get('group_updates',[]):
            if group.get('id') and group['id'] not in aliases:
                definitions[group['id']]=semantic_definition(group)
        clusters=answer.get('clusters',clusters)
        merge_groups(definitions,assignments,clusters,aliases,answer.get('group_merges',[]),core)
    return {'assignments':assignments,'groups':list(definitions.values()),'aliases':aliases,
            'clusters':clusters,'labeler':'DeepSeek high'}


def packet(bundle, pending, previous):
    anatomy=bundle['anatomy'];spectrum=bundle['spectrum'];aliases=previous.get('aliases',{})
    definitions={g['id']:g for g in spectrum.get('groups',[])}
    definitions.update({g['id']:g for g in previous.get('groups',[])})
    definitions={k:v for k,v in definitions.items() if canonical(k,aliases)==k}
    examples=collections.defaultdict(list)
    for point in anatomy['points']:
        label=(previous.get('assignments',{}).get(probe_key(point),{})
               if point.get('source_family')=='probe' or point.get('source')=='corpus:scan' else {})
        group=canonical(label.get('group',point.get('semantic_group',point['cluster'])),aliases)
        examples[group].append(point)
    groups=[]
    for group in definitions.values():
        all_members=examples[group['id']]
        ordered=sorted(all_members,key=lambda p:p['activation'])
        candidates=ordered[:2]+ordered[-2:]+sorted(ordered,key=lambda p:abs(p['activation']))[:1]
        members=list({p['id']:p for p in candidates}.values())
        groups.append(semantic_definition(group))
        groups[-1]['saved_examples']=[{'id':p['id'],'product':p['activation'],
            'response_pole':response_pole(p['activation']),
            'observed_prefix':p.get('observed_prefix',html.unescape(p['text']))} for p in members]
        groups[-1]['response_summary']=summary(p['activation'] for p in all_members)
        groups[-1]['n_observed_members']=len(all_members)
    points=[]
    for point in pending:
        prefix=point.get('observed_prefix',html.unescape(point['text']));full=point.get('input_text','')
        points.append({'id':point['id'],'observed_prefix':prefix,
            'following_prediction_target':full[len(prefix):] if full.startswith(prefix) else '',
            'anchor':{'position':point['position'],'token':html.unescape(point['peak_token']),
                      'gate':point['gate'],'value':point['up'],'product':point['activation'],
                      'response_pole':response_pole(point['activation'])}})
    # Auto-generated singleton fallbacks are not agent-supported cluster definitions.
    prior_clusters=[c for c in previous.get('clusters',[])
                    if not (len(c.get('member_groups',[]))==1
                            and c.get('id')=='group-'+c['member_groups'][0]
                            and not c.get('reason'))]
    return {'semantic_groups':groups,'prior_clusters':prior_clusters,
            'core_group_ids':[g['id'] for g in spectrum.get('groups',[])],
            'observed_responses':summary(p['activation'] for p in anatomy['points']),
            'labelled_points':[{'id':p['id'],'group':g,'product':p['activation'],
                                'response_pole':response_pole(p['activation'])}
                               for g,rows in examples.items() if g in definitions for p in rows],
            'aliases':aliases,
            'experimental_points':points}


def annotate(bundle, folder, previous=None, llm=None, batch_size=32):
    if llm is None:
        from blind_tests import call
        llm=call
    previous=previous or {};folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    probes=[p for p in bundle['anatomy']['points']
            if p.get('source_family')=='probe' or p.get('source')=='corpus:scan']
    assignments=copy.deepcopy(previous.get('assignments',{}));aliases=copy.deepcopy(previous.get('aliases',{}))
    for p in probes:
        if p.get('source')=='corpus:scan' and p.get('semantic_group'):
            assignments.setdefault(probe_key(p),{'id':p['id'],'group':p['semantic_group'],
                                                  'reason':'Reused saved natural-corpus assignment.'})
    pending=[p for p in probes if probe_key(p) not in assignments]
    evidence=packet(bundle,pending,previous);write(folder/'evidence.json',evidence)
    definitions={g['id']:semantic_definition(g) for g in evidence['semantic_groups']}
    clusters=copy.deepcopy(previous.get('clusters',[]));calls=0
    core={g['id'] for g in bundle['spectrum'].get('groups',[])}
    batches=(mixed_batches(pending,batch_size)
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
        context=packet(bundle,batch,{'groups':list(definitions.values()),'clusters':clusters,
                                    'assignments':assignments,'aliases':aliases})
        messages=[{'role':'system','content':SYSTEM},
                  {'role':'user','content':json.dumps(context,ensure_ascii=False)}]
        call_folder=call_root if len(batches)==1 else call_root/('batch-%03d'%index)
        call_folder.mkdir(parents=True,exist_ok=True)
        write(call_folder/'messages.json',messages)
        answer=llm(messages,call_folder/'call','anatomy-semantics',max_tokens=393216);calls+=1
        for group in answer.get('new_groups',[]) + answer.get('group_updates',[]):
            if group.get('id') and group['id'] not in aliases:
                definitions[group['id']]=semantic_definition(group)
        clusters=answer.get('clusters',clusters)
        merge_groups(definitions,assignments,clusters,aliases,answer.get('group_merges',[]),core)
        by_id={p['id']:p for p in probes if probe_key(p) in assignments}
        by_id.update({p['id']:p for p in batch})
        for row in answer.get('labels',[]):
            group=canonical(row.get('group'),aliases)
            if row.get('id') in by_id and group in definitions:
                assignments[probe_key(by_id[row['id']])]={**row,'group':group}
        call_records.append({'directory':str(call_folder),'requested':len(batch)})
        write(folder/'label-progress.json',{'completed_calls':calls,'total_calls':len(batches),
              'assigned':sum(probe_key(p) in assignments for p in probes),'requested':len(probes),
              'finished_at':now()})
        write(folder/'assignments-checkpoint.json',assignments)
    # Unassigned groups remain visible; there is no sign-based classification fallback.
    used=set();normal=[]
    for cluster in clusters:
        members=list(dict.fromkeys(canonical(g,aliases) for g in cluster.get('member_groups',[])))
        members=[g for g in members if g in definitions and g not in used]
        if not members:continue
        used.update(members);normal.append({**cluster,'member_groups':members})
    remaining=[identity for identity in definitions if identity not in used]
    if remaining:
        normal.append({'id':'unresolved-cluster-assignment','name':'Unresolved semantic grouping',
                       'pole':'either','member_groups':[], 'unassigned_groups':remaining})
    result={'groups':list(definitions.values()),'clusters':normal,'assignments':assignments,'aliases':aliases,
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
    memberships.update({g:c['id'] for c in semantics['clusters'] for g in c.get('unassigned_groups',[])})
    assignments=semantics['assignments'];quadrants=collections.defaultdict(collections.Counter)
    for point in anatomy['points']:
        if point.get('source_family')=='probe' or point.get('source')=='corpus:scan':
            label=assignments.get(probe_key(point),{})
            group=label.get('group',point.get('semantic_group',point['cluster']))
            if label:point['label_source']='saved-scan-label' if point.get('source')=='corpus:scan' else 'saved-probe-label';point['label_reason']=label.get('reason','')
        else:group=point.get('semantic_group',point['cluster'])
        group=canonical(group,semantics.get('aliases',{}))
        point['semantic_group']=group
        point['cluster']=memberships.get(group,group)
        q=('g+' if point['gate']>0 else 'g-')+('u+' if point['up']>0 else 'u-')
        quadrants[point['cluster']][q]+=1
    for point in anatomy['points']:
        point['response_pole']=response_pole(point['activation'])
    anatomy['semantic_groups']=[describe_group(g,[p['activation'] for p in anatomy['points']
        if p['semantic_group']==g['id']]) for g in semantics['groups']]
    anatomy['clusters']=[describe_group(g,[p['activation'] for p in anatomy['points']
        if p['cluster']==g['id']]) for g in semantics['clusters']]
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
