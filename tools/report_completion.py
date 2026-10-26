"""Complete report views from saved measurements and existing agent text."""
import bisect
import collections
import copy
import re
from pathlib import Path
from common import read


def coordinate(unit):
    return int(unit['layer']),int(unit.get('neuron',unit.get('index',unit.get('feature'))))


def corpus_counts(bundle):
    spectrum=bundle.get('spectrum')
    scan=bundle.get('bigscan') or {}
    values,counts=scan.get('hist_centers'),scan.get('hist')
    if not spectrum or values is None or counts is None:return
    edges=spectrum['edges'];totals=[0]*(len(edges)-1)
    for value,count in zip(values,counts):
        i=max(0,min(len(totals)-1,bisect.bisect_right(edges,value)-1))
        totals[i]+=int(count)
    spectrum['corpus_hist']={'counts':totals,'n_tokens':sum(totals),
        'source':'saved full-corpus product-value counts',
        'group_shares':'Selected labelled examples within each bin; not full-corpus semantic labels.'}


def minimal_pairs(bundle,folders,report=None):
    from pole_selection import select
    anatomy=bundle.setdefault('anatomy',{})
    if anatomy is None:anatomy={};bundle['anatomy']=anatomy
    # Tiny standalone fixtures can supply folders without a publication directory.
    if report is None:
        figures=[read(Path(f)/'figure.json') for f in folders if (Path(f)/'figure.json').exists()]
        context={**bundle,'custom':{'figures':[{**f,'kind':'sweep'} for f in figures]}}
        report=Path(folders[0]).parent if folders else Path('.')
    else:context=bundle
    selected=select(context,report,folders);pairs=[];sources=[]
    for pole,sign in [('positive',1),('negative',-1)]:
        item=selected.get(pole)
        if not item:continue
        rows=[r for r in item['rows'] if r.get('trace') and r.get('selected') and sign*r['selected']['product']>0]
        if not rows:continue
        row=max(rows,key=lambda r:sign*r['selected']['product']);trace=row['trace'];figure=item['figure']
        pairs.append({'text':row['text'],'tokens':[t['token'] for t in trace],
            'gate':[t['gate'] for t in trace],'up':[t['value'] for t in trace],
            'act':[t['product'] for t in trace],'source_sweep':figure['id'],
            'condition':row['condition'],'label':row.get('label',row.get('replacement','')),
            'pole':pole,'activation':row['selected']['product'],
            'anchor_index':row['selected'].get('position')})
        sources.append({'figure':figure['id'],'pole':pole,'template':figure.get('template'),
                        'score':item['request'].get('score')})
    if pairs:
        anatomy['headline']=pairs
        anatomy['headline_source']={'selection':'Representative measured positive and negative scored anchors; each retains its own replacement frame.',
                                   'sweeps':sources}


def unit_labels(bundle,folders):
    names={r['id']:r.get('name',r['id']) for r in bundle.get('routes',[])}
    weights={'upstream':collections.defaultdict(collections.Counter),
             'downstream':collections.defaultdict(collections.Counter)}
    for folder in folders:
        folder=Path(folder)
        if not (folder/'atlas-raw.json').exists() or not (folder/'route-factors.json').exists():continue
        assignments=read(folder/'route-factors.json')['assignments']
        for row in read(folder/'atlas-raw.json'):
            route=assignments.get(row.get('id'))
            if row.get('status')!='complete' or route not in names:continue
            for side in weights:
                for unit in row.get(side,[]):
                    weights[side][coordinate(unit)][route]+=abs(unit['weight'])
    explicit={}
    for field in ['circuit','family']:
        for side in ['upstream','downstream']:
            for unit in (bundle.get(field) or {}).get(side,[]):
                if unit.get('label'):explicit.setdefault(coordinate(unit),unit['label'])
    for unit in (bundle.get('family') or {}).get('family',[]):
        if unit.get('label'):explicit.setdefault(coordinate(unit),unit['label'])
    target=bundle.get('neuron') or {}
    if target.get('layer') is not None:
        explicit.setdefault(coordinate(target),bundle.get('claim',{}).get('title','This neuron'))
    def fill(unit,side=None):
        if unit.get('label'):return
        key=coordinate(unit)
        if key in explicit:unit['label']=explicit[key];return
        choices=([side] if side else ['upstream','downstream'])
        candidates=[(score,route,s) for s in choices for route,score in weights[s].get(key,{}).items()]
        if not candidates:return
        score,route,side=max(candidates)
        unit['label']=('Writer in ' if side=='upstream' else 'Reader in ')+names[route]
        unit['label_basis']={'route':route,'pooled_absolute_contribution':score,
            'method':'Largest pooled absolute contribution by named route in this measured atlas.'}
    for field in ['circuit','family']:
        for side in ['upstream','downstream']:
            for unit in (bundle.get(field) or {}).get(side,[]):fill(unit,side)
    for unit in (bundle.get('family') or {}).get('family',[]):fill(unit)
    for route in bundle.get('routes',[]):
        for unit in route.get('upstream',[]):fill(unit,'upstream')
    for cluster in bundle.get('clusters',[]):
        for unit in cluster.get('key_units',[]):fill(unit,'upstream')


def full_reading(bundle):
    saved=bundle.get('agent_read')
    if not saved:return
    full=copy.deepcopy(saved)
    full.setdefault('main_job',saved.get('one_line',''))
    if 'remainders' not in full:
        full['remainders']=[{'what':pole.title()+' pole: '+item['group'],
            'evidence':item.get('notes',''),'status':'unresolved'}
            for pole,item in saved.get('poles',{}).items()
            if item.get('group') in ['unresolved','unplaced','mixture']]
    full['text_source']='Existing reader output; the main-job line and unresolved pole notes are reused verbatim.'
    bundle['agent_read_full']=full


def complete(bundle,request,data=None):
    from anatomy_selection import build
    if request.get('report') and (Path(request['report'])/'report.json').is_file():
        build(bundle,data if data is not None else read(Path(request['report'])/'report.json'),request)
    from anatomy_semantics import apply, prior_labels
    apply(bundle,prior_labels(request))
    corpus_counts(bundle)
    minimal_pairs(bundle,request.get('sweeps',[]),request.get('report'))
    unit_labels(bundle,request.get('sections',[]))
    full_reading(bundle)
