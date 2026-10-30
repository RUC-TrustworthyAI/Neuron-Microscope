import contextlib
import datetime
import fcntl
import json
import os
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temp.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def union_seconds(intervals):
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, start if end is None else end))
        end = stop if end is None else max(end, stop)
    return total


@contextlib.contextmanager
def lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class Timeline:
    def __init__(self, root):
        self.root = Path(root)
        self.rows = read(self.root / 'timeline.json') if (self.root / 'timeline.json').exists() else []

    @contextlib.contextmanager
    def span(self, name, resource, stage):
        start = time.time()
        row = {'name': name, 'resource': resource, 'stage': stage, 'start': start}
        try:
            yield row
            row['status'] = 'complete'
        except BaseException as error:
            row.update(status='failed', error=str(error))
            raise
        finally:
            row.update(end=time.time(), seconds=time.time() - start)
            self.rows.append(row)
            write(self.root / 'timeline.json', self.rows)

    def status(self, phase, **extra):
        write(self.root / 'status.json', {'phase': phase, 'updated_at': now(), 'pid': os.getpid(), **extra})
