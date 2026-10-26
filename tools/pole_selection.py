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
