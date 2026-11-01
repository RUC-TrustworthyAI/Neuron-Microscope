import json
import math
import tempfile
import unittest
from pathlib import Path
from measured_response import response_pole, describe_group, mixed_batches


class MeasuredResponseTests(unittest.TestCase):
    def test_native_activation_sign_in_all_four_quadrants(self):
        values = [g/(1+math.exp(-g))*u for g,u in [(2,3),(2,-3),(-2,3),(-2,-3)]]
        self.assertEqual(list(map(response_pole,values)),['positive','negative','negative','positive'])
        self.assertEqual(response_pole(0),'zero')
        self.assertEqual(response_pole(None),'unknown')
        group=describe_group({'id':'x','pole':'negative'},[0.1,300,-50])
        self.assertEqual(group['pole'],'either')
        self.assertEqual(group['response_summary']['positive'],2)

    def test_each_scan_batch_spans_response_range_without_losing_points(self):
        points=[{'id':str(i),'activation':i-128} for i in range(256)]
        batches=mixed_batches(points,32)
        self.assertEqual(len(batches),8)
        self.assertEqual(sorted(p['id'] for b in batches for p in b),sorted(p['id'] for p in points))
        for batch in batches:
            self.assertEqual(len(batch),32)
            self.assertLess(min(p['activation'] for p in batch),-100)
            self.assertGreater(max(p['activation'] for p in batch),100)



if __name__ == '__main__': unittest.main()
