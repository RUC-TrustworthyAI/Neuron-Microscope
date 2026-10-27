"""Refresh spectrum labels from a saved explanation and cached blind predictions."""
import argparse
import copy
import os
import shutil
from pathlib import Path
from common import read,write,digest


def run(report,output,tokenizer_file,llm=None):
    from report import source_bundle,merge
    from blind_anatomy import prior,prediction_points,predict,apply_spectrum
    from tokenizers import Tokenizer
    report,output=Path(report),Path(output);output.mkdir(parents=True)
    bundle,_=source_bundle((report/'index.html').read_text())
    data=read(report/'report.json');previous=prior({'report':str(report)})
    version=(bundle.get('anatomy') or {}).get('explanation_version')
    latest=max((p for p in previous if p['version']==version),key=lambda p:p['created_at'])
    frozen={k:latest[k] for k in ('version','explanation','response_rules')}
    predictions=predict(prediction_points(bundle,data),data,frozen,output/'predictions',previous,llm=llm,
                        tokenizer=Tokenizer.from_file(str(tokenizer_file)))
    changed=copy.deepcopy(bundle);apply_spectrum(changed,predictions)
    prose=copy.deepcopy(bundle.get('section_prose') or {});prose.pop('spectrum',None)
    page,rendered=merge((report/'index.html').read_text(),{'spectrum':changed['spectrum'],'section_prose':prose})
    target=output/'report';shutil.copytree(report,target,copy_function=os.link)
    temporary=target/'index.html.tmp';temporary.write_text(page);temporary.replace(target/'index.html')
    write(target/'paper-data.json',{'bundle':rendered})
    archived=target/'artifacts'/('spectrum-predictions-'+predictions['version'][:12]);archived.mkdir()
    for name in ('response-predictions.json','frozen-explanation.json','result.json','tool-request.json'):
        shutil.copy2(output/'predictions'/name,archived/name)
    result={'status':'complete','html':str(target/'index.html'),'source_report':str(report),
            'source_html_sha256':digest(report/'index.html'),'new_model_forwards':0,
            'llm_requests':predictions['llm_requests'],'new_predictions':predictions['new_predictions'],
            'reused_predictions':predictions['reused_predictions'],'wall_seconds':predictions['wall_seconds'],
            'explanation_version':predictions['version'],'spectrum_points':len(changed['spectrum']['points'])}
    write(output/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--tokenizer',required=True,type=Path)
    args=parser.parse_args();print(__import__('json').dumps(run(args.report,args.output,args.tokenizer)))


if __name__=='__main__':main()
