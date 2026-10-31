"""Measure real throughput/memory before selecting full-campaign concurrency."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from .full_feature_viz import BatchRunner, config_A, DEFAULT_COLOR
from .five_variants import run_variant, write_json
from .models import PROJECT_ROOT


def available_mib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])/1024
    raise RuntimeError('Cannot read host memory availability')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--anchors',type=Path,required=True,help='Trusted mixed4c anchor .pt file')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    old_anchors=a.anchors
    anchors=torch.load(old_anchors,map_location='cpu',weights_only=True)
    # Same A settings and full schedule, even during a short numerical smoke test.
    cfg=config_A(8)
    runner=BatchRunner()
    runner.run(a.output/'equivalence/new','mixed4c',[447],'positive',cfg)
    if not (a.output/'equivalence/old/metrics.json').exists():
        run_variant(a.output/'equivalence/old',447,'positive',DEFAULT_COLOR,None,cfg)
    before=np.asarray(Image.open(a.output/'equivalence/old/final.png')).astype(int)
    after=np.asarray(Image.open(a.output/'equivalence/new/items/0/final.png')).astype(int)
    difference={'max_u8':int(np.abs(before-after).max()),'mean_u8':float(np.abs(before-after).mean())}
    if not (a.output/'equivalence/old_repeat/metrics.json').exists():
        run_variant(a.output/'equivalence/old_repeat',447,'positive',DEFAULT_COLOR,None,cfg)
    repeated=np.asarray(Image.open(a.output/'equivalence/old_repeat/final.png')).astype(int)
    difference['original_repeat_mean_u8']=float(np.abs(before-repeated).mean())
    difference['original_repeat_max_u8']=int(np.abs(before-repeated).max())
    difference['initial_equal']=bool(np.array_equal(np.asarray(Image.open(a.output/'equivalence/old/initial.png')),np.asarray(Image.open(a.output/'equivalence/new/items/0/initial.png'))))
    difference['check']='Exact initialization; final tolerance calibrated to measured original CUDA repeat variation, not bitwise equality'
    write_json(a.output/'equivalence.json',difference)
    assert difference['initial_equal'],difference
    assert difference['mean_u8']<=2*difference['original_repeat_mean_u8']+.1, difference
    assert difference['max_u8']<=2*difference['original_repeat_max_u8']+1,difference
    results=[];skipped=[]
    cases=[('conv2d0','positive',[1,8,16,32,64]),
           ('mixed4c','positive',[1,8,16,32,64]),
           ('mixed5b','positive',[1,8,16,32,64]),
           ('mixed4c','diversity',[1,4,8,16])]
    for layer,slot,sizes in cases:
        per_group=None
        for groups in sizes:
            free=available_mib()
            estimate=per_group*groups*1.25 if per_group else 500
            if estimate+2048>free:
                skipped.append({'layer':layer,'slot':slot,'groups':groups,'available_mib':free,'estimated_mib':estimate})
                continue
            # Duplicate known target only for throughput benchmarking; not production exports.
            channels=[447]*groups if layer=='mixed4c' else list(range(groups))
            cfg=config_A(a.steps);cfg['save_steps']=[a.steps]
            try:
                r=runner.run(a.output/f'{layer}_{slot}_b{groups}',layer,channels,slot,cfg,anchors if slot=='diversity' else None)
                results.append(r)
                per_group=r['peak_allocated_mib']/groups
            except torch.cuda.OutOfMemoryError:
                gc.collect();torch.cuda.empty_cache()
                skipped.append({'layer':layer,'slot':slot,'groups':groups,'reason':'CUDA OOM; stopped this batch size'})
                break
            write_json(a.output/'progress.json',{'results':results,'skipped':skipped,'equivalence':difference})
    write_json(a.output/'complete.json',{'results':results,'skipped':skipped,'equivalence':difference})


if __name__=='__main__':main()
