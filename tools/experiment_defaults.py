"""Host sampling policy; saved inputs retain every original token through the anchor."""
import copy
import hashlib
from common import read
from settings import load

def measured_plan(plan, report, seed=None, settings=None):
    cfg=copy.deepcopy(settings if settings is not None else load())
    seed=cfg['seed'] if seed is None else seed
    data=read(report/'report.json');chosen=[];seen=set();identities=set()
    def add(case, explicit=False):
        identity=str(case if isinstance(case,(str,int)) else case['case_id'])
        p=data['points'][identity];document=str(p['document'])
        if identity not in identities and (explicit or document not in seen):
            identities.add(identity);seen.add(document)
            metadata=case if isinstance(case,dict) else {}
            chosen.append({**metadata,'case_id':identity,'_document':document,'_product':p['values'][2]})
    for key in ['top_documents','bottom_documents']:
        for case in data['examples'].get(key,[])[:cfg['atlas']['extremes_per_pole']]:add(case)
    for band in data['bands']:
        for case in band['points'][:1]:
            if len(chosen)<cfg['atlas']['documents']:add(case)
    for key in ['top_documents','bottom_documents']:
        for case in data['examples'].get(key,[]):
            if len(chosen)<cfg['atlas']['documents']:add(case)
    for case in (plan.get('atlas') or {}).get('cases',[]):add(case,explicit=True)
    tests=set()
    for sign in [1,-1]:
        rows=sorted(list({r['_document']:r for r in chosen if r['_product']*sign>0}.values()),
                    key=lambda r:hashlib.sha256((str(seed)+r['_document']).encode()).hexdigest())
        if len(rows)>1:
            n=max(1,int(len(rows)*cfg['atlas']['holdout_fraction']))
            tests.update(r['_document'] for r in rows[:n])
    result=copy.deepcopy(plan)
    result['atlas']={**(result.get('atlas') or {}),**cfg['atlas'],
                    'cases':[{**{k:v for k,v in r.items() if not k.startswith('_')},'split':'test' if r['_document'] in tests else 'train'} for r in chosen]}
    # Joint display inputs are authored explicitly; natural atlas cases retain full prefixes.
    if result.get('family'):
        result['family']['neighbours_per_side']=cfg['family']['neighbours_per_side']
    return result

def semantic_cases(jobs, quota):
    """Representative saved cases from every currently named semantic group."""
    chosen=[]
    for job in jobs:
        path=__import__('pathlib').Path(job['request']['output'])/'sections.json'
        if not path.exists():continue
        spectrum=read(path).get('spectrum') or {}
        for group in spectrum.get('groups',[]):
            points=[p for p in spectrum.get('points',[]) if p.get('group')==group['id']]
            points.sort(key=lambda p:-abs(p['activation']))
            selected=[]
            for sign in [1,-1]:
                point=next((p for p in points if sign*p['activation']>0),None)
                if point is not None and point not in selected:selected.append(point)
            selected.extend(p for p in points if p not in selected)
            chosen.extend({'case_id':str(p['id']),'semantic_group':group['id']} for p in selected[:quota])
    return chosen
