"""Collect recorded interpretation costs and attach them after the agents finish."""
import json
import os
import shutil
from pathlib import Path
from common import read,write,now

STAGES={'prepare-evidence':0,'01-interpretation':1,'02-experiments':2,'03-revision':3}
LEGACY={'spectrum-labeler':2,'anatomy-semantics':2,'blind-writer':2,'paper-reader':3}
DEFINITIONS=[
 ('Prepare saved evidence','Fixed code · CPU + semantic labeler','Initial distributions, extrema, quantile examples, token statistics and saved gate/value/product; amplitude-stratified corpus scan and semantic labels before the initial explanation.'),
 ('Form an initial explanation','Agent 1','Initial title, claim, positive and negative responses, and what would change the interpretation.'),
 ('Design and measure experiments','Agent 2 + target model','Replacement figures, RelP graphs, activation search, semantic groups, anatomy, interventions, family traces, circuit measurements and blind tests.'),
 ('Revise and independently review','Agent 3 + independent reader','Revised explanation, Criticism, figure captions and independent assessment.'),
 ('Render the report','Fixed code · CPU','The generated interactive report, assembled from saved text and measurements.')
]

def collect(batch,unit):
    batch=Path(batch)
    rows=[{'step':i+1,'name':name,'resource':resource,'detail':detail,
           'api_usd':0.,'api_requests':0,'gpu_seconds':0.,'gpu_card_hours':0.}
          for i,(name,resource,detail) in enumerate(DEFINITIONS)]
    units=read(batch/'manifest.json')['units'];single=units==[unit]
    api_sources=[];unassigned=[]
    api_folders=list((batch/'logs/api').iterdir())
    for directory in read(batch/'manifest.json').get('additional_api_logs',[]):
        api_folders.extend(Path(directory).iterdir())
    for folder in sorted(api_folders):
        if not (folder/'attempt.json').is_file():continue
        attempt=read(folder/'attempt.json')
        parts=attempt.get('client_route','').strip('/').split('/')
        index=None
        if len(parts)>1 and parts[0]==unit:index=STAGES.get(parts[1])
        elif single and parts and parts[0] in LEGACY:index=LEGACY[parts[0]]
        if index is None:
            if parts and parts[0] not in units:unassigned.append(str(folder))
            continue
        row=rows[index];row['api_requests']+=1
        bill=read(folder/'billing.json') if (folder/'billing.json').exists() else {}
        row['api_usd']+=bill.get('estimated_usd',0.)
        api_sources.append(os.path.relpath(folder,batch))
    # Raw worker results can survive outside units/ after an explicitly resumed job.
    # Deduplicate copies by the worker finish time, allocation and experiment identities.
    seen=set();workers=[]
    marker='/units/'+unit+'/'
    for result_path in sorted(batch.rglob('batch-result.json')):
        result=read(result_path)
        if result.get('wall_seconds') is None:continue
        experiments=result.get('experiments',[])
        if not any(marker in str(item.get('output','')) for item in experiments):continue
        queue_path=result_path.parent/'queue.json'
        if not queue_path.is_file():continue
        queue=read(queue_path);devices=queue.get('gpu_pair',[])
        if not devices:continue
        key=(result.get('finished_at'),result['wall_seconds'],tuple(devices),
             tuple((item.get('id'),item.get('tool')) for item in experiments))
        if key in seen:continue
        seen.add(key)
        seconds=result['wall_seconds'];card_hours=seconds*len(devices)/3600
        rows[2]['gpu_card_hours']+=card_hours
        workers.append({'record':str(result_path.relative_to(batch)),
                        'worker_seconds':seconds,'gpu_count':len(devices),
                        'gpu_card_hours':card_hours})
    for row in rows:row['gpu_seconds']=row['gpu_card_hours']*3600/4
    return {'title':'Interpretation Costs','unit':unit,'completed_steps':5,'steps':rows,
        'total_api_usd':sum(row['api_usd'] for row in rows),
        'total_api_requests':sum(row['api_requests'] for row in rows),
        'total_gpu_card_hours':sum(row['gpu_card_hours'] for row in rows),
        'api_sources':api_sources,'worker_sources':workers,
        'unassigned_api_records':unassigned,'generated_at':now(),
        'basis':'Recorded worker runtime expressed as four-GPU equivalent time, including failed and resumed jobs. API costs are token-price estimates, not account invoices. Prior activation collection is excluded.'}

def finalize(batch,unit,handoff):
    """Final host-only step: no agent, API request or model forward."""
    from report import refresh_layout
    from stage_session import link_file
    batch,handoff=Path(batch),Path(handoff)
    workspace=handoff.parent/'workspace'
    number=1
    while (workspace/('paper-costs-%03d'%number)).exists():number+=1
    output=workspace/('paper-costs-%03d'%number)
    costs=collect(batch,unit)
    result=refresh_layout(handoff/'report',output,bundle_patch={'interpretation_costs':costs})
    write(output/'interpretation-costs.json',costs)
    staged=handoff/'report-with-costs'
    shutil.copytree(output,staged,copy_function=link_file)
    history=batch/'units'/unit/'publication-history'/('costs-%03d'%number)
    history.mkdir(parents=True)
    (handoff/'report').rename(history/'report')
    staged.rename(handoff/'report')
    write(workspace/'latest-paper.json',{'path':str(output/'index.html')})
    stage_result=read(workspace/'stage-result.json')
    stage_result['report']=str(output/'index.html');write(workspace/'stage-result.json',stage_result)
    write(batch/'units'/unit/'interpretation-costs.json',costs)
    return result
