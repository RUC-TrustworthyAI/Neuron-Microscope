"""Timing aggregation from synthetic records; no API or model calls."""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from report_costs import collect,interval_seconds,stream_interval

def iso(seconds):
    return datetime.fromtimestamp(1700000000+seconds,timezone.utc).isoformat()

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))

class CostTimingTests(unittest.TestCase):
    def test_overlap_is_elapsed_time_not_request_sum(self):
        self.assertEqual(interval_seconds([(10,20),(15,25),(11,12),(30,35)]),20)
        self.assertEqual(interval_seconds([]),0)

    def test_archived_streams_are_explicitly_estimated(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            save(folder/'headers.json',{'status':200})
            (folder/'response.sse').write_text('data: [DONE]\n')
            os.utime(folder/'headers.json',(100,100));os.utime(folder/'response.sse',(112,112))
            self.assertEqual(stream_interval(folder),((100,112),True))
            save(folder/'timing.json',{'first_chunk_at':iso(4),'last_chunk_at':iso(10)})
            interval,estimated=stream_interval(folder)
            self.assertEqual(interval[1]-interval[0],6)
            self.assertFalse(estimated)
            save(folder/'timing.json',{'first_chunk_at':None,'last_chunk_at':None})
            self.assertEqual(stream_interval(folder),(None,False))

    def test_per_neuron_totals_preserve_fees_and_ignore_other_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch=Path(tmp);unit='L1_N2'
            save(batch/'manifest.json',{'units':[unit,'L2_N3'],
                'additional_api_logs':[str(batch/'logs/api')]})
            save(batch/'units'/unit/'01-interpretation/preparation/initial-scan/preparation.json',{'started_at':iso(0)})
            save(batch/'units'/unit/'PIPELINE_TERMINAL.json',{'at':iso(120)})
            calls=[(unit,'02-experiments',10,20),(unit,'02-experiments',15,25),
                   (unit,'01-interpretation',18,30),('L2_N3','02-experiments',0,200)]
            for number,(neuron,stage,start,end) in enumerate(calls):
                folder=batch/'logs/api'/str(number)
                save(folder/'attempt.json',{'client_route':f'/{neuron}/{stage}/v1/messages','started_at':iso(start-1)})
                save(folder/'timing.json',{'first_chunk_at':iso(start),'last_chunk_at':iso(end)})
                save(folder/'billing.json',{'estimated_usd':.1})
            for number,(start,end) in enumerate([(100,110),(105,115)]):
                folder=batch/'units'/unit/'02-experiments/workspace/experiments'/str(number)
                save(folder/'batch-result.json',{'finished_at':iso(end),'wall_seconds':end-start,
                    'experiments':[{'id':str(number),'tool':'probe','output':str(folder/'probe')}]})
                save(folder/'queue.json',{'gpu_pair':[0,1,2,3]})
            result=collect(batch,unit)
            self.assertEqual(result['total_gpu_seconds'],15)
            self.assertEqual(result['steps'][2]['gpu_seconds'],20)
            self.assertEqual(result['total_api_stream_seconds'],20)
            self.assertEqual([row['api_stream_seconds'] for row in result['steps']],[0,12,15,0,0])
            self.assertEqual(result['total_elapsed_seconds'],120)
            self.assertAlmostEqual(result['total_api_usd'],.3)
            self.assertEqual(result['total_api_requests'],3)
            self.assertFalse(result['api_stream_time_estimated'])

if __name__=='__main__':unittest.main()
