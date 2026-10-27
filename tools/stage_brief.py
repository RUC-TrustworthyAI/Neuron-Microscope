"""One evidence overview, with full records available through the existing reader."""
import collections
from pathlib import Path
from common import read


def statistics(value):
    if not isinstance(value,dict):
        return {'count':len(value)} if isinstance(value,list) else value
    result={}
    for key,item in value.items():
        if item is None or isinstance(item,(str,int,float,bool)):
            result[key]=item
        elif isinstance(item,list):
            result[key]={'count':len(item)}
        elif isinstance(item,dict):
            scalars={k:v for k,v in item.items() if v is None or isinstance(v,(str,int,float,bool))}
            result[key]=scalars if scalars else {'count':len(item)}
    return result


def build(session,arguments):
    from report import source_bundle
    packet=session.prepare()
    current=session.root/'stage-result.json'
    saved=read(current) if current.exists() else session.prior
    report=Path(saved['report']).parent if current.exists() else Path(session.config['report'])
    bundle,_=source_bundle((report/'index.html').read_text())
    sections={}
    names={'distribution':'distribution','spectrum':'spectrum','search':'search',
           'clusters':'clusters','anatomy':'anatomy','ablation':'ablation',
           'family':'family','circuit':'circuit','tests':'tests'}
    prose=bundle.get('section_prose') or {}
    for name,key in names.items():
        value=bundle.get(key)
        entry={'available':value is not None,'statistics':statistics(value)}
        if name in prose:entry['introduction']=prose[name]
        sections[name]=entry
    contexts=packet['contexts_json']['cases']
    for key in ('token_selectivity','peak_census','gate_value','output_effects'):
        sections[key]={'available':True,'statistics':statistics(packet[key+'_json'])}
    interpretation=saved.get('interpretation',{})
    measurements=session.evidence({'section':'measurements'})
    for job in session.jobs:
        file=Path(job['request']['output'])/'sections.json'
        if not file.exists():continue
        for name,value in read(file).items():
            if name in sections:
                sections[name]['available']=True
                sections[name].setdefault('measurements',[]).append({
                    'experiment_id':job['request']['id'],'statistics':statistics(value)})
    feedback=session.evidence({'section':'response_predictions','limit':arguments.get('limit',4)})
    figures=bundle.get('custom') or {}
    figures=figures if isinstance(figures,list) else figures.get('figures',[])
    graphs=bundle.get('graphs_curated') or {}
    graphs=graphs if isinstance(graphs,list) else graphs.get('picks',[])
    return {'section':'brief','stage':session.stage,'neuron':session.config['neuron'],
            'interpretation':interpretation,'sections':sections,
            'contexts':{'count':len(contexts),
                        'groups':dict(collections.Counter(g for c in contexts for g in c['groups']))},
            'experiment_index':measurements['measurements'],
            'figures':[{k:f[k] for k in ('id','title','rationale','caption') if k in f} for f in figures],
            'graphs':[{k:g[k] for k in ('id','title','why','look_at') if k in g} for g in graphs],
            'cluster_definitions':bundle.get('cluster_definitions',[]),
            'response_predictions':feedback,
            'read_details':{
                'sections':'Use --sections SECTION ... to read several complete evidence sections in one call.',
                'contexts':'Use --section contexts --ids CASE_ID ... for complete observed prefixes and anchors.',
                'routes':'Use --section routes --ids GRAPH_ID ... for complete measured route members.',
                'measurements':'Use experiment_index paths and file names for complete saved measurements.',
                'response_predictions':'Use --section response_predictions with --ids or --offset and --limit for more cases.'}}
