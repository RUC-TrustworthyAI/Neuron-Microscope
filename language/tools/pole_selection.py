"""Select measured illustrations independently for each observed product sign."""
from pathlib import Path
from common import read


def sweeps(bundle, report, folders=()):
    custom=bundle.get('custom') or {}
    figures=custom if isinstance(custom,list) else custom.get('figures',[])
    by_id={f['id']:dict(f) for f in figures if f.get('kind')=='sweep'}
    locations={identity:Path(report)/'artifacts'/('sweeps-'+identity) for identity in by_id
               if (Path(report)/'artifacts'/('sweeps-'+identity)).exists()}
    for folder in list((Path(report)/'artifacts').glob('sweeps-*'))+list(map(Path,folders)):
        if (folder/'figure.json').exists():
            saved=read(folder/'figure.json');identity=saved['id']
            by_id.setdefault(identity,saved);locations[identity]=folder
    answer=[]
    for identity,figure in by_id.items():
        folder=locations.get(identity)
        request=read(folder/'tool-request.json') if folder and (folder/'tool-request.json').exists() else {}
        rows=read(folder/'conditions.json') if folder and (folder/'conditions.json').exists() else []
        if not rows and folder and (folder/'result.json').exists():rows=read(folder/'result.json').get('conditions',[])
        rows=[r for r in rows if r.get('status')=='complete']
        values=[r['selected']['product'] for r in rows if r.get('selected')] or [r['activation'] for r in figure.get('results',[])]
        if not values:continue
        pole=figure.get('pole') or request.get('pole')
        if pole in ['positive','negative']:poles=[pole]
        elif pole in ['both','either']:poles=['positive','negative']
        else:
            reduction=(request.get('score') or {}).get('reduction')
            if reduction=='max':poles=['positive']
            elif reduction=='min':poles=['negative']
            else:poles=[s for s,test in [('positive',max(values)>0),('negative',min(values)<0)] if test]
        answer.append({'figure':figure,'request':request,'rows':rows,'values':values,'poles':poles})
    return answer


def select(bundle, report, folders=()):
    choices=sweeps(bundle,report,folders)
    distribution=bundle.get('distribution') or {}
    selected={}
    for pole,sign,field in [('positive',1,'corpus_max'),('negative',-1,'corpus_min')]:
        extreme=distribution.get(field)
        eligible=[x for x in choices if pole in x['poles']]
        if extreme is not None and sign*extreme<=0:
            eligible=[x for x in eligible if any(sign*v>0 for v in x['values'])]
        if not eligible:continue
        def rank(item):
            values=item['values']
            reach=max(sign*v for v in values)
            spread=max(values)-min(values)
            return (reach>0,spread>0,min(max(0,reach),spread),reach,item['figure']['id'])
        selected[pole]=max(eligible,key=rank)
    return selected


def corpus_strips(data, preferred=()):
    points,documents=data.get('points',{}),data.get('documents',{})
    result={}
    for pole,sign,group in [('positive',1,'top_documents'),('negative',-1,'bottom_documents')]:
        candidates=list(dict.fromkeys([str(x) for x in preferred if x is not None]+[str(x) for x in data.get('examples',{}).get(group,[])]))
        case=next((c for c in candidates if c in points and sign*points[c]['values'][2]>0),None)
        if case is None:continue
        point=points[case];doc=documents[str(point['document'])];position=point['position']
        start,stop=max(0,position-24),min(len(doc['tokens']),position+11)
        result[pole]={'tokens':doc['tokens'][start:stop],'values':doc['product'][start:stop],
            'text':''.join(doc['tokens'][start:stop]),'case_id':case,'source_start':start,
            'anchor_index':position-start,'pole':pole,
            'caption':pole.title()+' saved context · product '+format(point['values'][2],'.6g')+'.'}
    return result


def comparison_ladder(bundle, report, folders=()):
    """One common-frame signed comparison, derived only from saved scored traces."""
    candidates=[]
    distribution=bundle.get('distribution') or {}
    positive=abs(distribution.get('corpus_max') or 1.)
    negative=abs(distribution.get('corpus_min') or 1.)
    for item in sweeps(bundle,report,folders):
        fillers=[]
        for row in item['rows']:
            trace=row.get('trace',[])
            positions=row.get('scored_positions',[])
            if not trace or not positions:continue
            peak=max(positions,key=lambda p:abs(trace[p]['product']))
            point=trace[peak]
            fillers.append({'tag':row.get('label',row.get('replacement','')),
                'text':row.get('replacement',''),'full_text':row['text'],
                'activation':point['product'],'peak_index':peak,'peak_token':point['token'],
                'tokens':[t['token'] for t in trace],'values':[t['product'] for t in trace],
                'source_condition':row['condition'],'scored_positions':positions})
        if len(fillers)<2:continue
        hi=max(f['activation'] for f in fillers);lo=min(f['activation'] for f in fillers)
        both=hi>0 and lo<0
        balanced=min(max(hi,0)/positive,max(-lo,0)/negative)
        rank=(both,balanced,max(hi,0)/positive+max(-lo,0)/negative)
        candidates.append((rank,{'template':item['figure'].get('template'),'fillers':fillers,
            'source_figure_id':item['figure']['id'],
            'metric':'signed_largest_absolute_product_at_saved_scored_positions',
            'caption':'Signed neuron activation at the largest absolute response within the recorded scoring span; every row uses the same template and scoring rule.'}))
    if not candidates:return None
    preferred=(bundle.get('display_selection') or {}).get('headline_figure_id')
    explicit=next((value for _,value in candidates if value['source_figure_id']==preferred),None)
    if explicit:return explicit
    best=max(candidates,key=lambda item:(len(item[1]['fillers'])==10,item[0]))
    if best[0][1]>=.1 or distribution.get('corpus_max',0)<=0 or distribution.get('corpus_min',0)>=0:
        return best[1]
    # A common frame that barely touches one observed pole must not replace that pole's evidence.
    pos=max(candidates,key=lambda item:max(f['activation'] for f in item[1]['fillers']))[1]
    neg=min(candidates,key=lambda item:min(f['activation'] for f in item[1]['fillers']))[1]
    if pos['source_figure_id']==neg['source_figure_id']:return best[1]
    fillers=[]
    for index,(frame,sign) in enumerate([(pos,1),(neg,-1)],1):
        ordered=sorted(frame['fillers'],key=lambda f:-sign*f['activation'])
        chosen=ordered[:4]+sorted(ordered,key=lambda f:abs(f['activation']))[:2]
        seen=set()
        for row in chosen:
            if row['source_condition'] in seen:continue
            seen.add(row['source_condition'])
            fillers.append({**row,'tag':str(row['tag'])+' ['+str(index)+']',
                'template':frame['template'],'source_figure_id':frame['source_figure_id']})
    return {'template':None,'fillers':fillers,'metric':best[1]['metric'],
        'frames':[{'source_figure_id':f['source_figure_id'],'template':f['template']} for f in [pos,neg]],
        'caption':'Signed peaks from two measured templates, marked [1] and [2]; each row uses its saved scoring span. Hover a row for its actual text. Complete comparisons remain in the figures below.'}
