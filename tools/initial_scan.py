"""Prepare private scan samples for explanation feedback and a final held-out test."""
import copy
import html
import hashlib
import time
from pathlib import Path
from common import read, write, now


def prepare(report, config, activations, folder, llm=None):
    if config.get('settings',{}).get('anatomy_scan',{}).get('enabled',True) is False:
        return None
    from report import source_bundle
    from spectrum import build as spectrum
    from anatomy_selection import build
    from anatomy_scan import ensure
    report,folder=Path(report),Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    # Host-only route metadata: the container never receives the archive or API records.
    write(folder/'host-session.json',{'stage':'prepare-evidence','neuron':config['neuron']})
    started=now();clock=time.perf_counter()
    source=(report/'index.html').read_text()
    bundle,_=source_bundle(source);data=read(report/'report.json')
    # Obtain the same numeric amplitude intervals without importing any previous labels.
    bundle['spectrum']=spectrum(data,[],{})
    bundle['spectrum']['groups']=[]
    request={'report':str(report),'activations':str(activations),'model':config['model'],
             'tokenizer':config.get('tokenizer',str(Path(config['model'])/'tokenizer.json')),
             'settings':config['settings'],'sections':[]}
    build(bundle,data,request)
    if ensure(request,bundle,folder) is None:return None
    build(bundle,data,request)
    points=[copy.deepcopy(p) for p in bundle['anatomy']['points'] if p.get('source')=='corpus:scan']
    splits=split_documents(points,set(map(str,data['documents'])))
    for point in points:
        point['evaluation_split']=splits[point['id']]
        point['cluster']='unassigned'
        for key in ('semantic_group','label_reason','label_source'):
            point.pop(key,None)
    scan=read(folder/'anatomy-scan.json')
    scan['points']=points
    scan['split_policy']='Document-disjoint feedback and final holdout; documents present in the initial report are feedback only.'
    write(folder/'selected-scan.json',scan)
    write(folder/'preparation.json',{'started_at':started,'finished_at':now(),
        'wall_seconds':time.perf_counter()-clock,'scan_points':len(points),
        'feedback_points':sum(p['evaluation_split']=='feedback' for p in points),
        'final_holdout_points':sum(p['evaluation_split']=='final_holdout' for p in points),
        'llm_requests':0,
        'new_model_forwards':0,'gpu_seconds':0})
    return folder


def split_documents(points, seen_documents):
    """Choose the final test without sharing documents with feedback or initial evidence."""
    groups={}
    for point in points:groups.setdefault(str(point['document']),[]).append(point)
    eligible=sorted((d for d in groups if d not in seen_documents),
                    key=lambda d:hashlib.sha256(('holdout:'+d).encode()).hexdigest())
    target=len(points)//2;chosen=set();count=0
    for document in eligible:
        if count>=target:break
        chosen.add(document);count+=len(groups[document])
    return {p['id']:('final_holdout' if str(p['document']) in chosen else 'feedback') for p in points}


def evidence(report, arguments):
    """Return semantic definitions and requested full-prefix scan contexts on demand."""
    saved=read(Path(report)/'anatomy-scan-evidence.json')
    points=saved['points']
    if arguments.get('ids'):
        ids=set(map(str,arguments['ids']));points=[p for p in points if p['id'] in ids]
    if arguments.get('group'):
        points=[p for p in points if arguments['group'] in (p.get('semantic_group'),p.get('cluster'))]
    offset=arguments.get('offset',0);limit=arguments.get('limit',4)
    page=points[offset:offset+limit]
    cases=[{'id':p['id'],'semantic_group':p.get('semantic_group'),'cluster':p['cluster'],
            'label_reason':p.get('label_reason',''),'observed_prefix':p['observed_prefix'],
            'anchor':{'position':p['position'],'token':html.unescape(p['peak_token']),
                      'gate':p['gate'],'value':p['up'],'product':p['activation'],
                      'response_pole':p.get('response_pole')}}
           for p in page]
    return {'section':'anatomy_scan','groups':saved['groups'],'clusters':saved['clusters'],
            'sampling':saved['sampling'],'total':len(points),'offset':offset,
            'next_offset':offset+len(page) if offset+len(page)<len(points) else None,'cases':cases}
