"""Host-owned GPU queue. File locks serialize jobs; active processes cause waiting."""
import contextlib
import fcntl
import os
import subprocess
import time
from pathlib import Path

class GPUPool:
    def __init__(self, pairs, lock_dir, poll_seconds=2, active=None):
        self.pairs = [tuple(x) for x in pairs]
        self.lock_dir = Path(lock_dir)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.poll_seconds = poll_seconds
        self.active = active or self._active

    @staticmethod
    def _active(pair):
        text = subprocess.check_output([
            'nvidia-smi', '--id='+','.join(map(str,pair)),
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        return bool(text.strip())

    @contextlib.contextmanager
    def acquire(self, timeout=1800):
        deadline = time.monotonic()+timeout
        while True:
            for pair in self.pairs:
                handle = (self.lock_dir/('pair-'+'-'.join(map(str,pair))+'.lock')).open('a+')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                try:
                    if self.active(pair):
                        handle.close()
                        continue
                except BaseException:
                    handle.close()
                    raise
                try:
                    yield pair
                finally:
                    handle.close()
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('GPU queue waiting time reached')
            time.sleep(self.poll_seconds)
