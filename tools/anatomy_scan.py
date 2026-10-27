"""CPU-only activation-amplitude sampling over one neuron's saved corpus."""
import bisect
import collections
import copy
import hashlib
import html
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
from common import read, write, now

POLICY = 'full-corpus-amplitude-stratified-v1'


def bin_index(value, edges):
    return max(0, min(len(edges)-2, bisect.bisect_right(edges, value)-1))


def proportional_counts(counts, total):
    """Allocate an exact total by candidate counts, using largest remainders."""
    population = sum(counts)
    total = min(total, population)
    if not population:return [0] * len(counts)
    quotas = [n * total // population for n in counts]
    order = sorted(range(len(counts)),
                   key=lambda i: (-(counts[i] * total % population), i))
    for i in order[:total-sum(quotas)]:quotas[i] += 1
    return quotas


def choose(points, edges, total=256, seed=42):
    """Retain scan points in proportion to their candidate-bin counts."""
    bins = collections.defaultdict(list)
    for p in points:
        bins[bin_index(p['activation'], edges)].append(p)
    quotas = proportional_counts([len(bins[i]) for i in range(len(edges)-1)], total)
    rng = np.random.default_rng(seed)
    selected = []
    counts = []
    for i, quota in enumerate(quotas):
        rows = sorted(bins[i], key=lambda p: str(p['id']))
        indices = np.sort(rng.choice(len(rows), quota, replace=False))
        selected.extend({**rows[int(j)], 'bin': i} for j in indices)
        counts.append({'i':i, 'lower':edges[i], 'upper':edges[i+1],
                       'candidates':len(rows), 'selected':len(indices)})
    return selected, counts


def saved(request):
    folders = sorted((Path(request['report'])/'artifacts').glob('*'))
    folders += list(map(Path, request.get('sections', [])))
    return [p/'anatomy-scan.json' for p in dict.fromkeys(folders)
            if (p/'anatomy-scan.json').is_file()]


def scan(request, bundle, output):
    """Read native BF16 channels once. Persist prefixes and token provenance."""
    output=Path(output);output.mkdir(parents=True, exist_ok=True)
    settings=request.get('settings',{})
    cfg=settings.get('anatomy_scan',{})
    limit=cfg.get('candidates_per_bin',cfg.get('points_per_bin',40));seed=settings.get('seed',42)
    root=Path(request['activations'])
    neuron=bundle['neuron']['id']
    edges=list(map(float,bundle['spectrum']['edges']))
    values=[p['activation'] for p in bundle['anatomy']['points']]
    while values and min(values)<edges[0]:edges.insert(0, edges[0]*2)
    while values and max(values)>edges[-1]:edges.append(edges[-1]*2)
    fingerprint=hashlib.sha256(json.dumps([str(root.resolve()),neuron,edges,limit,seed],
                            separators=(',',':')).encode()).hexdigest()
    for path in saved(request)+[output/'anatomy-scan.json']:
        if path.is_file() and read(path).get('fingerprint')==fingerprint:
            return path.parent
    started=now();clock=time.perf_counter()
    from dataset_reader import open_dataset
    api=open_dataset(root)
    try:
        hist=api.histogram(neuron,'product')
        bf16=(np.arange(65536,dtype=np.uint32)<<np.uint32(16)).view(np.float32)
        finite=np.isfinite(bf16)
        lookup=np.zeros(65536,dtype=np.intp)
        lookup[finite]=np.searchsorted(edges,bf16[finite],side='right')-1
        lookup=np.clip(lookup,0,len(edges)-2)
        totals=np.zeros(len(edges)-1,np.uint64)
        np.add.at(totals,lookup,hist)
        rng=np.random.default_rng(seed)
        ranks=[np.sort(rng.choice(int(n),min(limit,int(n)),replace=False))
               for n in totals]
        signature={'policy':POLICY,'neuron':neuron,'dataset':str(root),'seed':seed,
                   'edges':edges,'candidate_points_per_bin':limit,'fingerprint':fingerprint}
        write(output/'scan-request.json',signature)
        seen=np.zeros(len(totals),np.uint64);rows=[];tokens=0;shards=0
        read_clock=time.perf_counter()
        for start,values in api.iter_neuron(neuron):
            bits=(values[:,2].copy().view(np.uint32)>>np.uint32(16)).astype(np.uint16)
            bins=lookup[bits]
            counts=np.bincount(bins,minlength=len(totals)).astype(np.uint64)
            for i in range(len(totals)):
                lo=np.searchsorted(ranks[i],seen[i])
                hi=np.searchsorted(ranks[i],int(seen[i])+int(counts[i]))
                if hi<=lo:continue
                at=np.flatnonzero(bins==i)
                chosen=at[ranks[i][lo:hi]-int(seen[i])]
                for j in chosen:
                    rows.append((int(start+j),i,values[int(j)].tolist()))
            seen+=counts;tokens+=len(values);shards+=1
            write(output/'scan-progress.json',{'status':'scanning','tokens':tokens,
                  'total_tokens':api.manifest['tokens'],'shards':shards,
                  'elapsed_seconds':time.perf_counter()-clock})
        read_seconds=time.perf_counter()-read_clock
        # Known report positions already have semantic labels. Reuse those assignments.
        labels={str(p['id']):p['group'] for p in bundle['spectrum']['points']}
        data=read(Path(request['report'])/'report.json')
        mapping_file=Path(request['report'])/'dataset-index.json'
        mappings=read(mapping_file).get('points',{}) if mapping_file.exists() else {}
        known={}
        for key,p in data['points'].items():
            if key not in labels:continue
            mapped=mappings.get(str(key),{})
            index=mapped.get('dataset_token_index')
            if index is None:
                document=api.document(int(p['document']))
                index=int(document['token_start'])+int(p['position'])
            known[int(index)]={'case_id':str(key),'group':labels[str(key)]}
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(request.get('tokenizer') or Path(request['model'])/'tokenizer.json'))
        points=[]
        for index,band,channels in sorted(rows):
            doc_i=int(np.searchsorted(api.starts,index,side='right'))-1
            document=api.documents.slice(doc_i,1).to_pylist()[0]
            begin=int(document['token_start']);position=index-begin
            ids=np.array(api.token_ids[begin:index+1],copy=True).astype(int).tolist()
            reuse=known.get(index)
            point={'id':'scan-'+str(index),'gate':channels[0],'up':channels[1],
                   'activation':channels[2],'bin':band,
                   'peak_token':html.escape(tokenizer.decode(ids[-1:],skip_special_tokens=False)),
                   'text':html.escape(tokenizer.decode(ids[max(0,len(ids)-513):],skip_special_tokens=False)),
                   'observed_prefix':tokenizer.decode(ids,skip_special_tokens=False),
                   'token_ids':ids,'position':position,'token_id':ids[-1],
                   'source':'corpus:scan','source_family':'corpus',
                   'document':str(document['document_id']),'dataset_token_index':index,
                   'dataset':str(root),'neuron':neuron,
                   'cluster':reuse['group'] if reuse else 'unlabelled-scan',
                   'label_source':'saved-natural-label' if reuse else 'unlabelled',
                   'full_prefix_rule':'captured document start through anchor; no suffix'}
            if reuse:
                point.update(semantic_group=reuse['group'],case_id=reuse['case_id'])
            points.append(point)
        summary={**signature,'started_at':started,'finished_at':now(),
                 'scan_seconds':read_seconds,'total_seconds':time.perf_counter()-clock,
                 'tokens_scanned':tokens,'shards_scanned':shards,
                 'population_bin_counts':[int(x) for x in totals],
                 'sample_bin_counts':dict(collections.Counter(p['bin'] for p in points)),
                 'sampled_points':len(points),'gpu_seconds':0,'new_model_forwards':0,
                 'reused_semantic_labels':sum('semantic_group' in p for p in points)}
        write(output/'anatomy-scan.json',{**summary,'points':points})
        write(output/'scan-summary.json',summary)
        write(output/'scan-progress.json',{'status':'complete',**summary})
        return output
    finally:api.close()


def ensure(request,bundle,output):
    if not request.get('activations'):return None
    if request.get('settings',{}).get('anatomy_scan',{}).get('enabled',True) is False:return None
    folder=scan(request,bundle,output)
    output=Path(output)
    if folder.resolve()!=output.resolve():
        for name in ['anatomy-scan.json','scan-summary.json','scan-request.json','scan-progress.json']:
            if (folder/name).exists():shutil.copy2(folder/name,output/name)
        write(output/'scan-reuse.json',{'source':str(folder),'reused_at':now(),'new_scan_seconds':0})
        folder=output
    if str(folder) not in request.setdefault('sections',[]):request['sections'].append(str(folder))
    return folder
