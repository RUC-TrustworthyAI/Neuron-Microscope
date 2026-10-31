"""Bounded concurrent GPU trials with utilization, process memory and throughput."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from .benchmark_full_feature_viz import available_mib
from .five_variants import write_json
from .models import PROJECT_ROOT

HEADROOM_MIB = 3072


def projected_memory_mib(workers, batch):
    # Host measurements include CUDA contexts and CPU RSS, not just tensor peaks.
    # The original 350 MiB/context estimate understated unified-memory pressure.
    return workers * (1600 + batch * 105)


def stop_owned_children(children):
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def run_case(root,workers,batch,steps,layer):
    output=root/f'{layer}_w{workers}_b{batch}'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists():return json.loads((output/'complete.json').read_text())
    projected=projected_memory_mib(workers,batch)
    free=available_mib()
    if projected+HEADROOM_MIB>free:
        return {'workers':workers,'batch':batch,'skipped':'insufficient host headroom','free_mib':free,'projected_mib':projected}
    gpu_log=(output/'gpu.csv').open('w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,temperature.gpu',
                              '--format=csv,noheader,nounits','--loop-ms=500'],stdout=gpu_log,stderr=subprocess.DEVNULL)
    children=[];logs=[];memory=[];started=time.monotonic();stop_reason=None
    try:
        for i in range(workers):
            log=(output/f'worker{i}.log').open('w');logs.append(log)
            channels=list(range(i*batch,(i+1)*batch))
            args=[sys.executable,'-m','vision_repro.full_feature_viz','--output',str(output/f'worker{i}'),
                  '--layer',layer,'--steps',str(steps),'--channels',*map(str,channels)]
            children.append(subprocess.Popen(args,cwd=PROJECT_ROOT,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            free=available_mib()
            memory.append({'seconds':time.monotonic()-started,'available_mib':free})
            if free < 2048:
                stop_reason='Host MemAvailable below 2 GiB; terminated only owned benchmark children'
                stop_owned_children(children)
                break
            time.sleep(.5)
        elapsed=time.monotonic()-started
        codes=[p.returncode for p in children]
        if any(codes) and stop_reason is None:
            stop_reason=f'Benchmark worker failed: {codes}; inspect logs'
    finally:
        stop_owned_children(children)
        monitor.terminate();monitor.wait(timeout=10);gpu_log.close()
        for log in logs:log.close()
    write_json(output/'memory.json',memory)
    if stop_reason:
        result={'workers':workers,'batch':batch,'skipped':stop_reason,
                'min_available_mib':min(x['available_mib'] for x in memory)}
        write_json(output/'stopped.json',result)
        return result
    values=[];power=[]
    for line in (output/'gpu.csv').read_text().splitlines():
        fields=line.split(',')
        try:values.append(float(fields[1]));power.append(float(fields[3]))
        except (ValueError,IndexError):pass
    # Exclude model startup/tail for steady-state utilization, still retain raw CSV.
    steady=values[8:-4] if len(values)>16 else values
    runs=[json.loads((output/f'worker{i}/complete.json').read_text()) for i in range(workers)]
    result={'workers':workers,'batch_neurons_per_worker':batch,'simultaneous_images':workers*batch,
            'steps':steps,'elapsed_seconds':elapsed,'end_to_end_image_steps_per_second':workers*batch*steps/elapsed,
            'gpu_util_median':statistics.median(steady) if steady else None,'gpu_util_max':max(values) if values else None,
            'gpu_util_mean':statistics.mean(steady) if steady else None,'power_w_max':max(power) if power else None,
            'min_available_mib':min(x['available_mib'] for x in memory),
            'sum_peak_allocated_mib':sum(r['peak_allocated_mib'] for r in runs),
            'sum_peak_reserved_mib':sum(r['peak_reserved_mib'] for r in runs)}
    write_json(output/'memory.json',memory);write_json(output/'complete.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=256);p.add_argument('--layer',default='mixed5b')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);results=[]
    # Test one extra contender even if a single worker looks busy, to check throughput.
    # The earlier 2×32 and 3×20 trials did not improve throughput and exhausted
    # host headroom. Increase process count while keeping the image count modest.
    cases=[(1,32),(2,16),(3,12),(4,8)]
    for workers,batch in cases:
        result=run_case(a.output,workers,batch,a.steps,a.layer);results.append(result)
        write_json(a.output/'progress.json',{'results':results});print(json.dumps(result),flush=True)
        successful=[r for r in results if 'skipped' not in r and r['min_available_mib']>=2048]
        if workers>=2 and successful and result.get('gpu_util_median',0)>=98:
            break
    good=[r for r in results if 'skipped' not in r and r['min_available_mib']>=2048]
    winner=max(good,key=lambda r:r['end_to_end_image_steps_per_second']) if good else None
    write_json(a.output/'complete.json',{'results':results,'selected':winner})


if __name__=='__main__':main()
