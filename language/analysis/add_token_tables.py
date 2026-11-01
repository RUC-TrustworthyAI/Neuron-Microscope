"""Add six relative product-by-token tables to copies of saved neuron papers."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from tokenizers import Tokenizer

FRACTIONS = np.array([.1, .25, .5], dtype=np.float64)
LEVELS = (np.arange(65536, dtype=np.uint32)<<16).view(np.float32)
STATE = {}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def update_counts(ids, bits, counts, sums, squares, positive, negative, pt, nt):
    values = (np.asarray(bits, dtype=np.uint32)<<16).view(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite product in saved activations')
    size = len(counts)
    sums += np.bincount(ids, weights=values, minlength=size)
    squares += np.bincount(ids, weights=values.astype(np.float64)**2, minlength=size)
    for k in range(3):
        positive[k] += np.bincount(ids[values>pt[k]], minlength=size)
        negative[k] += np.bincount(ids[values<nt[k]], minlength=size)


def rankings(counts, sums, squares, positive, negative, pt, nt, labels):
    eligible = np.flatnonzero(counts>=50)
    means = sums[eligible]/counts[eligible]
    std = np.sqrt(np.maximum(squares[eligible]/counts[eligible]-means**2, 0))
    rows_by_id = {int(j): [int(j), labels[int(j)], int(counts[j]), round(float(mu),7), round(float(sd),7)] +
                  [int(x) for x in positive[:,j]]+[int(x) for x in negative[:,j]]
                  for j,mu,sd in zip(eligible,means,std)}
    result = []
    for side, matrix, thresholds, offset in [('positive',positive,pt,0),('negative',negative,nt,3)]:
        for k, fraction in enumerate(FRACTIONS):
            scores = matrix[k,eligible]/counts[eligible]
            order = np.lexsort((eligible, -counts[eligible], -scores))[:80]
            result.append({'view': {'key':side+'_'+str(k), 'mode':side, 'fraction':float(fraction),
                'threshold':float(thresholds[k]), 'column':5+offset+k},
                'minimum_count':50, 'rows':[rows_by_id[int(eligible[i])] for i in order]})
    return result


def insert_tables(source, tables, summary, css, js):
    if 'id="token-selectivity-data"' in source:
        raise ValueError('Source report already contains token tables')
    def script(name, obj):
        raw=json.dumps(obj, ensure_ascii=False, separators=(',',':'), allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
        return '<script id="'+name+'" type="application/json">'+raw+'</script>'
    addition=script('token-selectivity-data',tables)+script('token-selectivity-summary',summary)+'<script id="token-selectivity-renderer">'+js+'</script>'
    if source.count('</head>')!=1 or source.count('</body>')!=1:
        raise ValueError('Unexpected source HTML structure')
    return source.replace('</head>','<style id="token-selectivity-style">'+css+'</style>\n</head>',1).replace('</body>',addition+'\n</body>',1)


def initialize(dataset, reports, output, tokenizer):
    root=Path(dataset);manifest=read(root/'dataset.json')
    STATE.update(root=root, manifest=manifest, parts={p['id']:p for p in manifest['parts']},
        reports=Path(reports), output=Path(output),
        ids=np.load(root/manifest['corpus']['token_ids'], mmap_mode='r', allow_pickle=False),
        counts=np.load(Path(output)/'token-counts.i64.npy', mmap_mode='r', allow_pickle=False),
        labels=read(Path(output)/'token-labels.json'),
        css=Path(__file__).with_name('token_tables.css').read_text(),
        js=Path(__file__).with_name('token_tables.js').read_text())


def scan(unit):
    state=STATE;out=state['output'];name=unit['id'];source=state['reports']/name;dest=out/name
    begun=time.monotonic();job=out/'jobs'/(name+'.json');processed=0
    save(job,{'neuron':name,'status':'running','pid':os.getpid(),'processed_tokens':0,'started_at':utc()})
    try:
        part=state['parts'][unit['part']];column=unit['part_column'];root=state['root']/part['path']
        hfile=root/part['indices']['histogram'];hm=np.load(hfile,mmap_mode='r',allow_pickle=False)
        try:hist=np.array(hm[column,2],copy=True)
        finally:hm._mmap.close()
        n=state['manifest']['tokens']
        if int(hist.sum())!=n or np.any(hist[~np.isfinite(LEVELS)]):
            raise ValueError('Incomplete or non-finite source histogram')
        observed=LEVELS[hist>0];maximum=float(observed.max());minimum=float(observed.min())
        if not maximum>0>minimum:
            raise ValueError('Both signed poles are required for six relative tables')
        pt=FRACTIONS*maximum;nt=FRACTIONS*minimum;counts=state['counts'];vocab=len(counts)
        sums=np.zeros(vocab,np.float64);squares=np.zeros(vocab,np.float64)
        positive=np.zeros((3,vocab),np.int64);negative=np.zeros_like(positive)
        for j,shard in enumerate(part['shards']):
            if shard['token_start']!=processed:
                raise ValueError('Source token range has a gap or overlap')
            path=root/shard['path'];signature=(path.stat().st_size,path.stat().st_mtime_ns)
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            try:
                if list(array.shape)!=shard['shape'] or array.dtype!=np.uint16:
                    raise ValueError('Unexpected activation layout')
                start=shard['token_start'];end=start+shard['token_count']
                ids=np.asarray(state['ids'][start:end],dtype=np.int64)
                update_counts(ids,array[column,:,2],counts,sums,squares,positive,negative,pt,nt)
            finally:array._mmap.close()
            if signature!=(path.stat().st_size,path.stat().st_mtime_ns):
                raise ValueError('Source activation file changed during scan')
            processed=end
            if j%64==0 or processed==n:
                save(job,{'neuron':name,'status':'running','pid':os.getpid(),'processed_tokens':processed,
                          'tokens':n,'elapsed_s':round(time.monotonic()-begun,2),'shards_complete':j+1})
        if processed!=n:raise ValueError('Incomplete token coverage')
        checks=[]
        for k,f in enumerate(FRACTIONS):
            expected_p=int(hist[LEVELS>pt[k]].sum());expected_n=int(hist[LEVELS<nt[k]].sum())
            if int(positive[k].sum())!=expected_p or int(negative[k].sum())!=expected_n:
                raise ValueError('Threshold counts do not match the exact full-corpus histogram')
            if np.any(positive[k]>counts) or np.any(negative[k]>counts):raise ValueError('Token denominator exceeded')
            checks.append({'fraction':float(f),'positive_count':expected_p,'negative_count':expected_n})
        expected_sum=float(np.dot(hist.astype(np.float64),np.where(np.isfinite(LEVELS),LEVELS,0).astype(np.float64)))
        expected_square=float(np.dot(hist.astype(np.float64),np.where(np.isfinite(LEVELS),LEVELS,0).astype(np.float64)**2))
        if not np.isclose(sums.sum(),expected_sum,rtol=1e-9,atol=1e-4) or not np.isclose(squares.sum(),expected_square,rtol=1e-9,atol=1e-4):
            raise ValueError('Token moments do not match the full-corpus histogram')
        summary={'neuron':name,'tokens':n,'observed_token_ids':int(np.count_nonzero(counts)),
            'positive_max':maximum,'negative_min':minimum,'fractions':FRACTIONS.tolist(),
            'positive_thresholds':pt.tolist(),'negative_thresholds':nt.tolist(),
            'minimum_count':50,'ranking_size':80,'histogram_checks':checks,
            'comparisons':{'positive':'>','negative':'<'},'elapsed_s':time.monotonic()-begun}
        tables=rankings(counts,sums,squares,positive,negative,pt,nt,state['labels'])
        # Full copies keep the original report and every relative supporting link intact.
        shutil.copytree(source,dest,symlinks=False)
        np.savez(dest/'token_stats.npz',cnt=counts,summ=sums,sumsq=squares,fractions=FRACTIONS,
            positive_thresholds=pt,negative_thresholds=nt,positive_counts=positive,negative_counts=negative)
        save(dest/'token-rankings.json',tables);save(dest/'token-summary.json',summary)
        source_html=(source/'index.html').read_text()
        page=insert_tables(source_html,tables,summary,state['css'],state['js'])
        (dest/'index.html').write_text(page)
        save(dest/'token-tables-complete.json',{'complete':True,'neuron':name,'tokens':n,'tables':6,
             'rows_per_table':80,'minimum_count':50,'source_report':str(source),'finished_at':utc()})
        save(job,{'neuron':name,'status':'complete','processed_tokens':n,'elapsed_s':time.monotonic()-begun,'finished_at':utc()})
        return {'neuron':name,'elapsed_s':time.monotonic()-begun}
    except BaseException:
        failure={'neuron':name,'status':'failed','processed_tokens':processed,'error':traceback.format_exc(),'at':utc()}
        save(job,failure)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--reports',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);(args.output/'jobs').mkdir()
    began=time.monotonic();m=read(args.dataset/'dataset.json');units=read(args.dataset/m['neurons'])
    if not m['complete'] or not read(args.dataset/'verification.json')['complete']:raise ValueError('Dataset is incomplete')
    catalog=read(args.reports/'catalog.json');entries={r['id']:r for r in catalog['neurons']}
    if {u['id'] for u in units}!=set(entries):raise ValueError('Dataset and report neuron sets differ')
    if any(not entries[u['id']]['maximum']>0>entries[u['id']]['minimum'] for u in units):
        raise ValueError('A report lacks signed poles; inspect before starting')
    manifest={'dataset':str(args.dataset),'source_reports':str(args.reports),'neurons':len(units),
        'tokens_per_neuron':m['tokens'],'workers':args.workers,'fractions':FRACTIONS.tolist(),
        'minimum_count':50,'rows_per_table':80,'llm_calls':0,'new_model_forwards':0,'started_at':utc()}
    save(args.output/'manifest.json',manifest)
    for name in ['add_token_tables.py','token_tables.css','token_tables.js']:
        shutil.copy2(Path(__file__).with_name(name),args.output/name)
    save(args.output/'progress.json',{'status':'preparing_token_counts','neurons':len(units),'complete':0})
    tok=Tokenizer.from_file(str(args.tokenizer));vocab=tok.get_vocab_size(with_added_tokens=True)
    ids=np.load(args.dataset/m['corpus']['token_ids'],mmap_mode='r',allow_pickle=False)
    if len(ids)!=m['tokens']:raise ValueError('Token corpus length differs')
    counts=np.zeros(vocab,np.int64)
    for start in range(0,len(ids),10_000_000):
        block=np.asarray(ids[start:start+10_000_000],dtype=np.int64)
        if block.max()>=vocab:raise ValueError('Token ID exceeds tokenizer vocabulary')
        counts+=np.bincount(block,minlength=vocab)
    ids._mmap.close()
    if int(counts.sum())!=m['tokens']:raise ValueError('Token denominator differs')
    np.save(args.output/'token-counts.i64.npy',counts)
    labels=['']*vocab
    for token in np.flatnonzero(counts):labels[int(token)]=tok.decode([int(token)],skip_special_tokens=False)
    save(args.output/'token-labels.json',labels)
    units.sort(key=lambda u:(u['layer'],u['neuron']))
    completed=[];failures={}
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
            initargs=(str(args.dataset),str(args.reports),str(args.output),str(args.tokenizer))) as pool:
        futures={pool.submit(scan,u):u['id'] for u in units}
        while futures:
            done,_=wait(futures,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                name=futures.pop(future)
                try:
                    item=future.result();completed.append(name);print(json.dumps(item),flush=True)
                except BaseException as exc:
                    failures[name]=str(exc);print(json.dumps({'failed':name,'error':str(exc)}),flush=True)
            jobs=[read(f) for f in (args.output/'jobs').glob('*.json')]
            elapsed=time.monotonic()-began
            processed=sum(j.get('processed_tokens',0) for j in jobs)
            progress={'status':'scanning','neurons':len(units),'complete':len(completed),'failed':len(failures),
                'running':sum(j['status']=='running' for j in jobs),'workers':args.workers,
                'neuron_tokens_processed':processed,'neuron_tokens_total':len(units)*m['tokens'],
                'elapsed_s':round(elapsed,1),'estimated_remaining_s':round(elapsed*(len(units)-len(completed))/len(completed),1) if completed else None,
                'updated_at':utc()}
            save(args.output/'progress.json',progress)
    if failures:
        save(args.output/'failure.json',{'complete':False,'completed':len(completed),'failures':failures})
        save(args.output/'progress.json',{'status':'failed','neurons':len(units),'complete':len(completed),'failed':len(failures),'updated_at':utc()})
        raise SystemExit(1)
    if len(completed)!=len(units):raise ValueError('Incomplete neuron coverage')
    for name in completed:
        if not read(args.output/name/'token-tables-complete.json')['complete']:raise ValueError('Missing completion marker')
    shutil.copy2(args.reports/'index.html',args.output/'index.html');save(args.output/'catalog.json',catalog)
    save(args.output/'complete.json',{'complete':True,'neurons':len(completed),'tables':len(completed)*6,
        'tokens_per_neuron':m['tokens'],'elapsed_s':time.monotonic()-began,'llm_calls':0,'new_model_forwards':0})
    save(args.output/'progress.json',{'status':'complete','neurons':len(units),'complete':len(completed),
        'failed':0,'elapsed_s':time.monotonic()-began,'updated_at':utc()})


if __name__=='__main__':
    main()
