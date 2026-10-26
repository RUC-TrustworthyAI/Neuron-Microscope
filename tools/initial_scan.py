"""Prepare labelled corpus-scan evidence before the interpretation container starts."""
import copy
import html
import time
from pathlib import Path
from common import read, write, now


def prepare(report, config, activations, folder, llm=None):
    if config.get('settings',{}).get('anatomy_scan',{}).get('enabled',True) is False:
        return None
    from report import source_bundle, merge
    from spectrum import build as spectrum
    from anatomy_selection import build
    from anatomy_scan import ensure
    from anatomy_semantics import annotate, apply, checkpoint_labels
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
    for point in bundle['anatomy']['points']:
        point.pop('semantic_group',None)
        point['cluster']='unlabelled-scan' if point.get('source')=='corpus:scan' else 'unlabelled'
    previous=checkpoint_labels(folder)
    semantics=annotate(bundle,folder,previous,llm,
        batch_size=config['settings'].get('anatomy_scan',{}).get('label_batch_size',32))
    apply(bundle,semantics)
    points=[copy.deepcopy(p) for p in bundle['anatomy']['points'] if p.get('source')=='corpus:scan']
    selection=bundle['anatomy']['selection']
    evidence={'groups':semantics['groups'],'clusters':semantics['clusters'],
              'sampling':{k:selection[k] for k in ('policy','edges','bins','scan_anchors','scan_seconds','tokens_scanned','final_rule') if k in selection},
              'points':points}
    write(report/'anatomy-scan-evidence.json',evidence)
    page,_=merge(source,{'anatomy':bundle['anatomy']})
    (report/'index.html').write_text(page)
    write(folder/'preparation.json',{'started_at':started,'finished_at':now(),
        'wall_seconds':time.perf_counter()-clock,'scan_points':len(points),
        'new_model_forwards':0,'gpu_seconds':0})
    return folder


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
                      'gate':p['gate'],'value':p['up'],'product':p['activation']}}
           for p in page]
    return {'section':'anatomy_scan','groups':saved['groups'],'clusters':saved['clusters'],
            'sampling':saved['sampling'],'total':len(points),'offset':offset,
            'next_offset':offset+len(page) if offset+len(page)<len(points) else None,'cases':cases}
