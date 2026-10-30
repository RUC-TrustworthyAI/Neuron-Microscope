"""Refresh spectrum labels from a saved explanation and cached blind predictions."""
import argparse
import copy
import os
import shutil
from pathlib import Path
from common import read,write,digest


def run(report,output,tokenizer_file,llm=None,contexts=(),prediction_sources=(),fresh=False,resume_from=None):
    from report import source_bundle,merge
    from blind_anatomy import prior,prediction_points,predict,apply
    from tokenizers import Tokenizer
    report,output=Path(report),Path(output);output.mkdir(parents=True)
    bundle,_=source_bundle((report/'index.html').read_text())
    data=read(report/'report.json');previous=prior({'report':str(report),'sections':prediction_sources})
    version=(bundle.get('anatomy') or {}).get('explanation_version')
    latest=max((p for p in previous if version is None or p['version']==version),key=lambda p:p['created_at'])
    frozen={k:latest[k] for k in ('version','explanation','response_rules')}
    reused=() if fresh else previous
    if resume_from:
        reused=[{**frozen,'assignments':read(Path(resume_from)/'predictions-checkpoint.json')}]
    predictions=predict(prediction_points(bundle,data),data,frozen,output/'predictions',reused,llm=llm,
                        tokenizer=Tokenizer.from_file(str(tokenizer_file)))
    changed=copy.deepcopy(bundle);apply(changed,predictions)
    patches={k:changed[k] for k in ('spectrum','anatomy') if k in changed}
    # The spectrum's existing narrative can contain statistics for the previous labels.
    # Its standard introduction remains applicable after refreshing the labels.
    if 'spectrum' in (changed.get('section_prose') or {}):
        prose=copy.deepcopy(changed['section_prose']);prose.pop('spectrum')
        patches['section_prose']=prose
    page,rendered=merge((report/'index.html').read_text(),patches)
    target=output/'report';shutil.copytree(report,target,copy_function=os.link)
    temporary=target/'index.html.tmp';temporary.write_text(page);temporary.replace(target/'index.html')
    write(target/'paper-data.json',{'bundle':rendered})
    from spectrum_sampling import register_contexts
    register_contexts(data,contexts)
    write(target/'report.json',data)
    archived=target/'artifacts'/('spectrum-predictions-'+predictions['version'][:12]);archived.mkdir(exist_ok=True)
    for name in ('response-predictions.json','frozen-explanation.json','result.json','tool-request.json','sections.json'):
        temporary=archived/(name+'.tmp')
        shutil.copy2(output/'predictions'/name,temporary);temporary.replace(archived/name)
    result={'status':'complete','html':str(target/'index.html'),'source_report':str(report),
            'source_html_sha256':digest(report/'index.html'),'new_model_forwards':0,
            'llm_requests':predictions['llm_requests'],'new_predictions':predictions['new_predictions'],
            'reused_predictions':predictions['reused_predictions'],'missing_predictions':predictions['missing_predictions'],'wall_seconds':predictions['wall_seconds'],
            'explanation_version':predictions['version'],'spectrum_points':len(changed['spectrum']['points']),'anatomy_points':len(changed['anatomy']['points'])}
    write(output/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--tokenizer',required=True,type=Path)
    parser.add_argument('--fresh',action='store_true',help='Predict all saved points using the existing explanation.')
    parser.add_argument('--resume-from',type=Path,help='Resume saved predictions from an interrupted labeling operation.')
    args=parser.parse_args();print(__import__('json').dumps(run(args.report,args.output,args.tokenizer,fresh=args.fresh,resume_from=args.resume_from)))


if __name__=='__main__':main()
