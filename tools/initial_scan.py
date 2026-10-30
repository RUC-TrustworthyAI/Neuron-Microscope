"""Prepare the shared natural-corpus samples before interpretation."""
import html
from pathlib import Path
from common import read, write


def prepare(report, config, activations, folder, llm=None):
    if not config.get('settings', {}).get('spectrum_sampling', {}).get('enabled', True):
        return None
    from spectrum_sampling import prepare as prepare_spectrum
    folder=Path(folder)
    sampled=prepare_spectrum(report,config,activations,folder.parent/'spectrum-sample')
    timing=read(sampled/'sampling.json')
    folder.mkdir(parents=True,exist_ok=True)
    write(folder/'preparation.json',{k:timing[k] for k in
        ('started_at','finished_at','wall_seconds','sampled_points','llm_requests','new_model_forwards','gpu_seconds')})
    return sampled


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
