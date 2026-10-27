"""Update display prose in a report copy without review, predictions or inference."""
import copy
from pathlib import Path
from common import read,write,now


def merge_annotations(previous,current):
    result=copy.deepcopy(previous)
    for key,value in current.items():
        if isinstance(value,dict) and isinstance(result.get(key),dict):
            result[key]=merge_annotations(result[key],value)
        else:result[key]=copy.deepcopy(value)
    return result


def patch(bundle,annotations):
    result={}
    for name,field,items,prose in [
        ('figures','custom','figures',('title','rationale','caption')),
        ('graphs','graphs_curated','picks',('title','why','look_at'))]:
        if name not in annotations:continue
        data=copy.deepcopy(bundle.get(field) or {})
        rows=data if isinstance(data,list) else data.get(items,[])
        for row in rows:
            update=annotations[name].get(row['id'],{})
            row.update({k:update[k] for k in prose if k in update})
        result[field]=data
    if annotations.get('sections'):
        prose=copy.deepcopy(bundle.get('section_prose') or {})
        for section,value in annotations['sections'].items():
            prose[section]={**prose.get(section,{}),
                **{k:value[k] for k in ('summary','strip_caption') if k in value}}
        result['section_prose']=prose
    selection={k:annotations[k] for k in ('headline_figure_id','graph_ids') if k in annotations}
    if selection:result['display_selection']={**bundle.get('display_selection',{}),**selection}
    return result


def render(session,arguments):
    from report import source_bundle,refresh_layout
    saved=read(session.root/'stage-result.json')
    source=Path(saved['report']).parent
    bundle,_=source_bundle((source/'index.html').read_text())
    annotations=arguments.get('annotations',{})
    # Only display prose and selections are consumed; claim and rules stay fixed.
    supported={k:annotations[k] for k in ('figures','graphs','sections','headline_figure_id','graph_ids')
               if k in annotations}
    previous=read(source/'annotations.json') if (source/'annotations.json').exists() else {}
    merged=merge_annotations(previous,supported)
    number=1
    while (session.root/('paper-render-%03d'%number)).exists():number+=1
    output=session.root/('paper-render-%03d'%number)
    result=refresh_layout(source,output,bundle_patch=patch(bundle,supported))
    write(output/'annotations.json',merged)
    saved['report']=str(output/'index.html')
    saved.setdefault('agent_output',{})['annotations']=merged
    saved['finished_at']=now()
    write(session.root/'stage-result.json',saved)
    write(session.root/'latest-paper.json',{'path':saved['report']})
    return {**result,'report':str(output.relative_to(session.root)/'index.html')}
