"""Own-process supervisor and unified-memory telemetry for a fixed campaign."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from .benchmark_full_feature_viz import available_mib
from .five_variants import write_json
from .full_feature_campaign import status
from .models import PROJECT_ROOT
from .profile_full_concurrency import projected_memory_mib, stop_owned_children


def supervise(root, workers):
    root = root.resolve()
    lock = (root/'supervisor.lock').open('a')
    # Hold an exclusive lock for this queue's supervisor.
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((root/'protocol.json').read_text())
    batch = protocol['batch_images']
    children = {}; logs = []; launches = 0; failures = 0; pressure_stops = 0
    next_launch = 0.; stopping = False
    def request_stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        with (root/'telemetry.jsonl').open('a', buffering=1) as telemetry:
            while not stopping:
                for pid, child in list(children.items()):
                    code = child.poll()
                    if code is not None:
                        del children[pid]
                        if code != 0:
                            failures += 1
                progress = status(root)
                if progress['batches'].get('failed', 0) >= 3 or failures >= 3:
                    raise RuntimeError('Repeated worker failures; inspect errors before resuming')
                if not children and not progress['batches'].get('pending') and not progress['batches'].get('running'):
                    write_json(root/'supervisor_complete.json', progress)
                    break
                free = available_mib()
                if free < 2048 and children:
                    pid = next(reversed(children))
                    child = children.pop(pid)
                    stop_owned_children([child])
                    pressure_stops += 1
                    next_launch = time.monotonic()+60
                    # Reduce this supervisor's worker count after repeated memory pressure.
                    if pressure_stops >= 2:
                        workers = max(1, workers-1)
                elif len(children) < workers and time.monotonic() >= next_launch:
                    # Conservative incremental allocation after earlier workers warmed up.
                    if free >= projected_memory_mib(1, batch)+3072:
                        log = (root/f'worker_{launches:03d}.log').open('a'); logs.append(log)
                        child = subprocess.Popen([sys.executable, '-m', 'vision_repro.full_feature_campaign',
                                                  'worker', '--root', str(root)], cwd=PROJECT_ROOT,
                                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                        children[child.pid] = child
                        launches += 1
                        next_launch = time.monotonic()+20
                try:
                    gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,power.draw,temperature.gpu',
                                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5).stdout.strip()
                except subprocess.TimeoutExpired:
                    gpu = 'telemetry timeout'
                snapshot = {'time':time.time(), 'available_mib':free, 'gpu_csv':gpu,
                            'worker_pids':list(children), 'requested_workers':workers,
                            'launches':launches, 'pressure_stops':pressure_stops, 'failures':failures,
                            'progress':progress}
                telemetry.write(json.dumps(snapshot)+'\n')
                write_json(root/'supervisor.json', snapshot)
                time.sleep(5)
    finally:
        stop_owned_children(list(children.values()))
        for log in logs:
            log.close()
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--workers', type=int, required=True)
    a = p.parse_args()
    if a.workers < 1:
        p.error('At least one worker is required')
    supervise(a.root, a.workers)


if __name__ == '__main__':
    main()
