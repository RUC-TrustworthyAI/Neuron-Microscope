"""Refresh only Inside the neuron from a saved report and full-corpus scan."""
import argparse
import copy
import time
from pathlib import Path
from common import read,write,now
from report import source_bundle,refresh_layout
from anatomy_selection import build
from anatomy_scan import ensure
from anatomy_semantics import annotate,apply,probe_key,prior_labels
from settings import load


def run(request,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    started=now();clock=time.perf_counter()
    bundle,_=source_bundle((Path(request['report'])/'index.html').read_text())
    original=copy.deepcopy(bundle['anatomy'])
    previous=prior_labels(request)
    previous.setdefault('groups',original.get('semantic_groups',bundle['spectrum']['groups']))
    previous.setdefault('clusters',original.get('clusters',[]))
    assignments=previous.setdefault('assignments',{})
    for p in original.get('points',[]):
        if p.get('source_family')=='probe' and p.get('semantic_group'):
            assignments.setdefault(probe_key(p),{'id':p['id'],'group':p['semantic_group'],
                                    'reason':p.get('label_reason','Reused saved assignment.')})
    data=read(Path(request['report'])/'report.json')
    build(bundle,data,request)
    folder=output/'scan-and-labels'
    ensure(request,bundle,folder)
    build(bundle,data,request)
    labels_started=time.perf_counter()
    semantics=annotate(bundle,folder,previous,
                      batch_size=request['settings'].get('anatomy_scan',{}).get('label_batch_size',32))
    label_seconds=time.perf_counter()-labels_started
    apply(bundle,semantics)
    write(output/'request.json',request)
    published=output/'report'
    refresh_layout(request['report'],published,bundle_patch={'anatomy':bundle['anatomy']})
    archive(published,folder)
    scan=read(folder/'scan-summary.json') if (folder/'scan-summary.json').exists() else read(Path(bundle['anatomy']['selection']['scan_source']).parent/'scan-summary.json')
    result={'status':'complete','started_at':started,'finished_at':now(),
            'wall_seconds':time.perf_counter()-clock,'scan_seconds':scan['scan_seconds'],
            'scan_total_seconds':scan['total_seconds'],'label_wall_seconds':label_seconds,
            'tokens_scanned':scan['tokens_scanned'],'scan_candidates':scan['sampled_points'],
            'label_result':read(folder/'result.json'),
            'final_points':bundle['anatomy']['n'],
            'source_counts':dict(__import__('collections').Counter(p['source'].split(':')[0] for p in bundle['anatomy']['points'])),
            'bin_counts':bundle['anatomy']['selection']['bins'],
            'gpu_seconds':0,'new_model_forwards':0,'html':str(published/'index.html')}
    write(output/'summary.json',result)
    return result


def archive(report,folder):
    import os,shutil
    target=Path(report)/'artifacts'/'anatomy-scan-labels'
    if not target.exists():shutil.copytree(folder,target,copy_function=os.link)
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True,type=Path)
    parser.add_argument('--activations',required=True,type=Path)
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    request={'report':str(args.report.resolve()),'activations':str(args.activations.resolve()),
             'model':str(args.model.resolve()),'settings':load(),'sections':[]}
    print(__import__('json').dumps(run(request,args.output),ensure_ascii=False))

if __name__=='__main__':main()
