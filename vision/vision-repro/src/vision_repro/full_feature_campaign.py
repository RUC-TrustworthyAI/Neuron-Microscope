"""Resumable queue for all 5,804 neurons × five visualization modes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback

import torch
from .all_neuron_examples import LAYERS
from .benchmark_full_feature_viz import available_mib
from .five_variants import SLOTS, sha256, write_json
from .full_feature_viz import BatchRunner, config_A


def database(root):
    con=sqlite3.connect(root/'queue.sqlite',timeout=60)
    con.row_factory=sqlite3.Row
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, layer TEXT, slot TEXT, channels TEXT, state TEXT, pid INTEGER, error TEXT, updated REAL)')
    return con


def status(root):
    con=database(root)
    counts=Counter();jobs=Counter()
    for row in con.execute('SELECT channels,state FROM jobs'):
        jobs[row['state']]+=1
        if row['state']!='split':counts[row['state']]+=len(json.loads(row['channels']))
    con.close()
    result={'updated_at':datetime.now(timezone.utc).isoformat(),'neuron_variants':dict(counts),
            'batches':dict(jobs),'total_requested':5804*5,'steps':1024}
    temporary=root/f'.progress-{os.getpid()}.json'
    write_json(temporary,result)
    temporary.replace(root/'progress.json')
    return result


def initialize(root, batch_images=32):
    if not (root/'anchors/complete.json').exists():
        raise RuntimeError('All-layer natural anchors are not complete yet')
    con=database(root)
    if con.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]:
        con.close();return
    if batch_images<=0 or batch_images%4:raise ValueError('Batch images must be a positive multiple of four')
    protocol={'created_at':datetime.now(timezone.utc).isoformat(),'steps':1024,'config':config_A(),
              'neurons':5804,'variants':list(SLOTS),'requested_neuron_variants':29020,
              'physical_images':46432,'batch_images':batch_images,
              'single_image_neurons_per_batch':batch_images,'diversity_neurons_per_batch':batch_images//4,
              'anchor_protocol_sha256':sha256(root/'anchors/protocol.json'),
              'source_sha256':{name:sha256(Path(__file__).with_name(name)) for name in
                              ['full_feature_campaign.py','full_feature_viz.py','full_path_anchors.py']},
              'publish':{'policy':'Export only completed, validated batches. Uncomputed targets remain placeholders; no automatic hosting or upload.'},
              'memory_policy':'Leave 2 GiB MemAvailable headroom and reduce batch size under memory pressure.',
              'resume':'Per-batch Adam/FFT/RNG checkpoints at 256/512/768/1024; completed batches skipped',
              'web_delivery':'Use the static export command; large datasets and PNGs remain separate from Git source.'}
    write_json(root/'protocol.json',protocol)
    ordered=sorted(LAYERS,key=lambda x:0 if x[0]=='mixed4c' else 1)
    unavailable=[]
    for layer,c,_ in ordered:
        anchors=torch.load(root/f'anchors/{layer}.pt',map_location='cpu',weights_only=True)
        for first in range(0,c,batch_images):
            for slot in SLOTS:
                members=4 if slot=='diversity' else 1
                chunk=batch_images//members
                for lo in range(first,min(c,first+batch_images),chunk):
                    channels=list(range(lo,min(c,lo+chunk,first+batch_images)))
                    needs_anchor=slot in ('path_positive','path_negative','diversity')
                    pole='negative' if slot=='path_negative' else 'positive'
                    good=[]
                    for ch in channels:
                        usable=not needs_anchor or anchors[f'{ch}_{pole}']['usable']
                        if usable:good.append(ch)
                        else:
                            jid=f'{layer}_{slot}_unavailable_{ch}'
                            con.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)',(jid,layer,slot,json.dumps([ch]),'unavailable',None,'No nondegenerate natural anchor/score scale in cached 50k subset',time.time()))
                            unavailable.append([layer,ch,slot])
                    if good:
                        jid=f'{layer}_{slot}_{lo:04d}_{len(good):02d}'
                        con.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)',(jid,layer,slot,json.dumps(good),'pending',None,None,time.time()))
    con.commit();con.close()
    write_json(root/'unavailable.json',unavailable)
    print(json.dumps(status(root)),flush=True)


def worker(root, max_jobs=None):
    con=database(root)
    protocol=json.loads((root/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():
        if sha256(Path(__file__).with_name(name))!=digest:
            raise RuntimeError(f'Source changed after campaign initialization: {name}')
    runner=BatchRunner();anchors={};completed=0
    # Recover only jobs whose original campaign process is demonstrably gone.
    for row in con.execute("SELECT id,pid FROM jobs WHERE state='running'").fetchall():
        proc=Path(f"/proc/{row['pid']}/cmdline")
        try:command=proc.read_bytes()
        except FileNotFoundError:command=b''
        if b'full_feature_campaign' not in command or str(root).encode() not in command:
            con.execute("UPDATE jobs SET state='pending',pid=NULL WHERE id=?",(row['id'],))
    con.commit()
    while max_jobs is None or completed<max_jobs:
        if available_mib()<2300:
            write_json(root/'resource_wait.json',{'available_mib':available_mib(),'time':time.time()})
            time.sleep(30);continue
        con.execute('BEGIN IMMEDIATE')
        row=con.execute("SELECT * FROM jobs WHERE state='pending' ORDER BY rowid LIMIT 1").fetchone()
        if row is None:
            con.commit();break
        con.execute("UPDATE jobs SET state='running',pid=?,updated=? WHERE id=?",(os.getpid(),time.time(),row['id']))
        con.commit()
        channels=json.loads(row['channels']);members=4 if row['slot']=='diversity' else 1
        expected_mib=350+len(channels)*members*100
        split_reason=None
        if expected_mib+2048>available_mib() and len(channels)>1:
            split_reason='Reduced batch to preserve host memory headroom'
        else:
            try:
                if row['layer'] not in anchors:
                    anchors[row['layer']]=torch.load(root/f"anchors/{row['layer']}.pt",map_location='cpu',weights_only=True)
                folder=root/'batches'/row['id']
                result=runner.run(folder,row['layer'],channels,row['slot'],protocol['config'],anchors[row['layer']])
                con.execute("UPDATE jobs SET state='done',pid=NULL,updated=? WHERE id=?",(time.time(),row['id']))
                completed+=1
            except torch.cuda.OutOfMemoryError:
                gc.collect();torch.cuda.empty_cache()
                if len(channels)>1:split_reason='CUDA OOM; retry smaller independent batches'
                else:
                    con.execute("UPDATE jobs SET state='failed',error=?,pid=NULL,updated=? WHERE id=?",('CUDA OOM at one neuron',time.time(),row['id']))
            except Exception:
                error=traceback.format_exc()
                con.execute("UPDATE jobs SET state='failed',error=?,pid=NULL,updated=? WHERE id=?",(error,time.time(),row['id']))
                print(error,flush=True)
        if split_reason:
            middle=len(channels)//2
            for suffix,subset in [('a',channels[:middle]),('b',channels[middle:])]:
                con.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)',(row['id']+'_'+suffix,row['layer'],row['slot'],json.dumps(subset),'pending',None,None,time.time()))
            con.execute("UPDATE jobs SET state='split',error=?,pid=NULL,updated=? WHERE id=?",(split_reason,time.time(),row['id']))
        con.commit();print(json.dumps(status(root)),flush=True)
        if con.execute("SELECT COUNT(*) FROM jobs WHERE state='failed'").fetchone()[0]>=3:
            raise RuntimeError('Queue has at least three failed batches; inspect errors before resuming')
    result=status(root)
    if not result['batches'].get('pending') and not result['batches'].get('running'):
        write_json(root/'queue_finished.json',result)
    con.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['init','worker','status'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--batch-images',type=int,default=32)
    p.add_argument('--max-jobs',type=int)
    a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    if a.action=='init':initialize(root,a.batch_images)
    elif a.action=='worker':worker(root,a.max_jobs)
    else:print(json.dumps(status(root)),flush=True)


if __name__=='__main__':main()
