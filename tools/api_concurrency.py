"""Host-wide request slots shared by meters using the same API account."""
import fcntl
import time
from pathlib import Path


class RequestSlots:
    def __init__(self,directory,limit=10):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.limit=limit

    def acquire(self):
        while True:
            for index in range(self.limit):
                stream=(self.directory/('%03d.lock'%index)).open('a')
                try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:stream.close();continue
                return stream
            time.sleep(.1)

    @staticmethod
    def release(stream):
        try:fcntl.flock(stream,fcntl.LOCK_UN)
        finally:stream.close()
