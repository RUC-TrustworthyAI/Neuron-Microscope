"""Build the gate/up view from saved, labelled corpus samples."""
import collections
import copy
import html
from pathlib import Path
from common import read

POLICY='selected-corpus-and-measured-probes-v1'
SCAN_POLICY='selected-corpus-probes-and-proportional-scan-v2'
SHARED_POLICY='shared-spectrum-corpus'
UNLABELLED='unlabelled-probes'


def build(bundle,data,request):
    spectrum=bundle.get('spectrum') or {}
    if not spectrum.get('points') or not data.get('points'):return
    anatomy=copy.deepcopy(bundle.get('anatomy') or {})
    labels={str(p['id']):p['group'] for p in spectrum['points']}
    definitions={g['id']:g for g in spectrum.get('groups',[])}
    keep=set()
    for ids in data.get('examples',{}).values():keep.update(map(str,ids))
    for band in data.get('bands',[]):keep.update(map(str,band['points']))
    # Retain explicit choices made by this run's agents, including background controls.
    for group in list(spectrum.get('groups',[]))+list(anatomy.get('groups',[])):
        keep.update(map(str,group.get('case_ids',[])))
    if anatomy.get('strip',{}).get('case_id'):keep.add(str(anatomy['strip']['case_id']))
    for row in anatomy.get('pairs',[]):
        if row.get('case_id'):keep.add(str(row['case_id']))
    old={str(p['id']):p for p in anatomy.get('points',[]) if str(p['id']) in data['points']}
    shared=[p for p in spectrum['points'] if p.get('source_family')=='corpus'
            and p.get('observed_prefix') is not None
            and all(k in p for k in ('gate','up','activation','document','position'))]
    points=[]
    occupied=set()
    for sample in shared:
        location=(str(sample['document']),sample['position'])
        if location in occupied:continue
        occupied.add(location)
        row={k:copy.deepcopy(sample[k]) for k in ('id','document','position','dataset_token_index',
            'gate','up','activation','peak_token','observed_prefix','evaluation_split',
            'predicted_pole','prediction_reason','explanation_version','label_source') if k in sample}
        row.update(source='corpus:spectrum',source_family='corpus',cluster=sample.get('group','unlabelled'),
            text=html.escape(sample['observed_prefix']))
        points.append(row)
    if shared:
        quadrants=collections.defaultdict(collections.Counter)
        for point in points:
            quadrant=('g+' if point['gate']>0 else 'g-')+('u+' if point['up']>0 else 'u-')
            quadrants[point['cluster']][quadrant]+=1
        anatomy.update(points=points,n=len(points),clusters=copy.deepcopy(spectrum.get('groups',[])),
            semantic_groups=copy.deepcopy(spectrum.get('groups',[])),
            quadrants={k:dict(v) for k,v in quadrants.items()},
            group_names={g['id']:g['name'] for g in spectrum.get('groups',[])},
            colour_mode='predicted_pole',explanation_version=spectrum.get('explanation_version'))
        anatomy['selection']={'policy':SHARED_POLICY,'corpus_anchors':len(points),
            'corpus_rule':'Use the spectrum natural-corpus points once per document position.',
            'labels':'Both views reuse the saved predictions from the fixed initial explanation.',
            'raw_channels':'Raw pre-SiLU gate, up/value and native product.'}
        bundle['anatomy']=anatomy
        return
    for identity in sorted(keep,key=int):
        if identity not in data['points']:continue
        native=data['points'][identity];doc=data['documents'][native['document']];pos=native['position']
        location=(str(native['document']),pos)
        if location in occupied:continue
        occupied.add(location)
        row=copy.deepcopy(old.get(identity,{}))
        row.update(id=identity,gate=native['values'][0],up=native['values'][1],activation=native['values'][2],
            peak_token=html.escape(doc['tokens'][pos]),
            text=html.escape(''.join(doc['tokens'][max(0,pos-512):pos+1])),
            cluster=labels.get(identity,'unlabelled'),source='corpus:selected',source_family='corpus',
            case_id=identity,document=native['document'],position=pos)
        points.append(row)
    corpus_n=len(points);seen={};sources=[];excluded_blind=0
    report=Path(request['report'])
    folders=list(map(Path,request.get('sections',[])+request.get('sweeps',[])))
    folders+=sorted((report/'artifacts').glob('*'))
    folders=list(dict.fromkeys(folders))
    def add(row,source,source_file,role,experiment):
        if row.get('status')!='complete' or not row.get('selected'):return
        p=row['selected'];ids=row.get('token_ids')
        if ids is None:return
        key=(tuple(ids),p['position'],p['gate'],p['value'],p['product'])
        record={'file':str(source_file),'source':source,'role':role,'experiment':experiment}
        if key in seen:
            seen[key]['sources'].append(record);return
        offset=p.get('offset')
        end=offset[1] if offset else len(row.get('text',''))
        group=row.get('semantic_group') or p.get('semantic_group')
        if group not in definitions:group=UNLABELLED
        point={'id':'probe-%04d'%(len(seen)+1),'gate':p['gate'],'up':p['value'],
            'activation':p['product'],'peak_token':html.escape(p.get('token','')),
            'text':html.escape(row.get('text','')[:end]),'cluster':group,
            'source':source,'source_family':'probe','role':role,'experiment':experiment,
            'position':p['position'],'token_id':p.get('token_id'),'token_ids':ids,
            'input_text':row.get('text',''),'label_source':'existing' if group!=UNLABELLED else 'unlabelled',
            'sources':[record]}
        seen[key]=point;points.append(point)
    for folder in folders:
        raw=folder/'search-raw.json'
        if raw.exists():
            sources.append(str(raw))
            for region in read(raw):
                if region.get('id','').startswith('blind-'):
                    excluded_blind+=len(region.get('candidates',[]))+len(region.get('controls',[]));continue
                for role in ['candidates','controls']:
                    for row in region.get(role,[]):
                        add(row,'search:'+region['id'],raw,role,region.get('name',region['id']))
        conditions=folder/'conditions.json'
        if conditions.exists() and (folder/'figure.json').exists():
            figure=read(folder/'figure.json');sources.append(str(conditions))
            for row in read(conditions):add(row,'replacement:'+figure['id'],conditions,'replacement',figure.get('title',figure['id']))
    from anatomy_scan import saved, choose
    scan_files=saved(request)
    scan_counts=[]
    candidate_counts=None
    if scan_files:
        scan=read(scan_files[-1])
        occupied={(str(p.get('document')),p.get('position')) for p in points if p.get('source_family')=='corpus'}
        scan_candidates=[]
        for row in scan['points']:
            location=(str(row.get('document')),row.get('position'))
            if row.get('case_id') in keep or location in occupied:continue
            scan_candidates.append(copy.deepcopy(row))
            occupied.add(location)
        candidate_counts=dict(collections.Counter(p['source'].split(':')[0] for p in points+scan_candidates))
        scan_total=request.get('settings',{}).get('anatomy_scan',{}).get('total_points',256)
        retained,scan_counts=choose(scan_candidates,scan['edges'],scan_total,scan['seed'])
        points.extend(retained)
    quadrants=collections.defaultdict(collections.Counter)
    for point in points:
        quadrant=('g+' if point['gate']>0 else 'g-')+('u+' if point['up']>0 else 'u-')
        quadrants[point['cluster']][quadrant]+=1
    anatomy.update(points=points,n=len(points),quadrants={k:dict(v) for k,v in quadrants.items()},
        group_names={**{g['id']:g['name'] for g in spectrum.get('groups',[])},UNLABELLED:'Unlabelled experimental contexts'})
    anatomy['selection']={'policy':POLICY,'corpus_anchors':corpus_n,'probe_anchors':len(seen),
        'source_counts':dict(collections.Counter(p['source'].split(':')[0] for p in points)),
        'uniform_only_excluded':len(set(map(str,data.get('scatter',[])))-keep),
        'corpus_rule':'Union of saved token/document extrema, quantile anchors and explicit agent-selected cases.',
        'probe_rule':'Existing scored anchors from searches, matched controls and replacement conditions; identical input IDs/position/native values are deduplicated.',
        'labels':'Reuse existing semantic assignments only; no inference from sign, search hypothesis or official labels.',
        'independent_blind_tests':'Retained in Tests, excluded from the explanatory scatter.',
        'source_files':sources,'raw_channels':'Raw pre-SiLU gate, up/value and native product; no recomputation.',
        'reference_scope':'Matches the published mix of selected corpus and measured probe evidence; the author did not publish exact selection quotas.'}
    if scan_files:
        anatomy['selection'].update(
            policy=SCAN_POLICY,scan_anchors=sum(p['source']=='corpus:scan' for p in points),
            corpus_anchors=sum(p['source']=='corpus:selected' for p in points),
            probe_anchors=sum(p.get('source_family')=='probe' for p in points),
            source_counts=dict(collections.Counter(p['source'].split(':')[0] for p in points)),
            candidate_source_counts=candidate_counts,edges=scan['edges'],bins=scan_counts,
            scan_points_limit=scan_total,scan_candidate_points_per_bin=scan.get('candidate_points_per_bin',scan.get('points_per_bin')),
            scan_source=str(scan_files[-1]),
            scan_seconds=scan['scan_seconds'],tokens_scanned=scan['tokens_scanned'],
            corpus_rule='Saved extrema and quantile anchors plus uniformly sampled ranks in each full-corpus activation-amplitude bin.',
            final_rule=f'Retain up to {scan_total} scan points in proportion to scan candidate counts per amplitude bin, with largest-remainder rounding. Keep selected corpus and measured experimental anchors separately; no second mixed-pool sample.')
    bundle['anatomy']=anatomy
