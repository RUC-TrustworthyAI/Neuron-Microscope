import concurrent.futures
import tempfile
import threading
import time
import unittest
from api_concurrency import RequestSlots


class RequestSlotTests(unittest.TestCase):
    def test_separate_meters_share_slots_and_release_after_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            meters=[RequestSlots(tmp,2),RequestSlots(tmp,2)]
            active=0;peak=0;lock=threading.Lock();entered=threading.Barrier(2,timeout=3)
            def request(i):
                nonlocal active,peak
                meter=meters[i%2];slot=meter.acquire()
                try:
                    with lock:active+=1;peak=max(peak,active)
                    if i<2:entered.wait()
                    time.sleep(.02)
                    if i==1:raise RuntimeError('upstream error')
                finally:
                    with lock:active-=1
                    meter.release(slot)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                jobs=[pool.submit(request,i) for i in range(8)]
                failures=0
                for job in jobs:
                    try:job.result(timeout=5)
                    except RuntimeError:failures+=1
            self.assertEqual(peak,2);self.assertEqual(active,0);self.assertEqual(failures,1)
            slot=meters[0].acquire();meters[0].release(slot)


if __name__=='__main__':unittest.main()
