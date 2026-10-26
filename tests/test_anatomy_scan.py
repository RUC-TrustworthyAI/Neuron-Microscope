import unittest
import json,tempfile
from pathlib import Path
from anatomy_scan import choose,bin_index,proportional_counts

class AnatomyScanTests(unittest.TestCase):
    def test_exact_total_preserves_candidate_proportions_and_measurements(self):
        points=[{'id':str(i),'activation':(-0.2 if i<80 else 2.0),
                 'gate':i/8,'up':-i/4,'source':'corpus:scan'}
                for i in range(320)]
        selected,bins=choose(points,[-1,0,4],256,42)
        again,_=choose(list(reversed(points)),[-1,0,4],256,42)
        self.assertEqual([p['id'] for p in selected],[p['id'] for p in again])
        self.assertEqual([b['selected'] for b in bins],[64,192])
        self.assertEqual(len({p['id'] for p in selected}),256)
        originals={p['id']:p for p in points}
        for p in selected:
            self.assertEqual({k:p[k] for k in originals[p['id']]},originals[p['id']])
        self.assertEqual(bin_index(0,[-1,0,4]),1)
        self.assertEqual(bin_index(4,[-1,0,4]),1)

    def test_256_over_twelve_equal_candidate_bins(self):
        self.assertEqual(proportional_counts([40]*12,256),[22]*4+[21]*8)
        self.assertEqual(proportional_counts([0,2,1],256),[0,2,1])
        self.assertEqual(proportional_counts([0,0],256),[0,0])

    def test_saved_anchors_are_not_thinned_with_scan_candidates(self):
        from anatomy_selection import build
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'saved';folder.mkdir()
            data={'points':{str(i):{'document':'d','position':i,'values':[1,1,1]}
                            for i in range(80)},
                  'documents':{'d':{'tokens':['a']*80}},
                  'examples':{'top_tokens':[str(i) for i in range(80)]}}
            bundle={'spectrum':{'groups':[{'id':'g','name':'saved'}],
                                'points':[{'id':str(i),'group':'g'} for i in range(80)]},
                    'anatomy':{}}
            scan={'edges':[-1,0,2],'seed':42,'policy':'saved-candidates',
                  'points_per_bin':40,'scan_seconds':0,'tokens_scanned':1000,
                  'points':[{'id':'scan-'+str(i),'document':'scan-doc','position':i,
                             'activation':-0.5 if i<40 else 1.0,'gate':1,'up':1,
                             'cluster':'g','source':'corpus:scan','source_family':'corpus'}
                            for i in range(80)]}
            (folder/'anatomy-scan.json').write_text(json.dumps(scan))
            build(bundle,data,{'report':str(root),'sections':[str(folder)],
                               'settings':{'anatomy_scan':{'total_points':64}}})
            selection=bundle['anatomy']['selection']
            self.assertEqual(selection['corpus_anchors'],80)
            self.assertEqual(selection['scan_anchors'],64)
            self.assertEqual(bundle['anatomy']['n'],144)

if __name__=='__main__':unittest.main()
