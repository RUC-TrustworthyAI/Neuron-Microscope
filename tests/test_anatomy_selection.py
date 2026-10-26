import copy,json,tempfile,unittest
from pathlib import Path
from anatomy_selection import build
from spectrum import anatomy_labels

class AnatomySelectionTests(unittest.TestCase):
    def test_saved_evidence_sources_and_semantic_labels_remain_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);art=p/'search';art.mkdir()
            data={'points':{str(i):{'document':'d','position':i,'values':[i,-i,-i*i]} for i in range(4)},
                  'documents':{'d':{'tokens':['a',' b',' c',' d']}},
                  'examples':{'top_tokens':['0'],'bottom_tokens':['1']},
                  'bands':[{'points':['2']}],'scatter':['0','1','2','3']}
            def measured(text,token,product):return {'status':'complete','text':text,'token_ids':[token],
                'selected':{'position':0,'gate':2.,'value':product/2,'product':product,'token':text,'token_id':token,'offset':[0,len(text)]}}
            a=measured('candidate',10,-4);control=measured('control',11,3)
            (art/'search-raw.json').write_text(json.dumps([
                {'id':'r','name':'Proposed negative group','pole':'negative','candidates':[a,a],'controls':[control]},
                {'id':'blind-positive','candidates':[measured('blind',12,10)]}]))
            b={'spectrum':{'groups':[{'id':'own','name':'Own saved label','pole':'negative'}],
                           'points':[{'id':str(i),'group':'own'} for i in range(4)]},'anatomy':{}}
            original=copy.deepcopy(data)
            build(b,data,{'report':str(p),'sections':[str(art)]})
            anatomy_labels(b)
            points=b['anatomy']['points']
            self.assertEqual(len(points),5)
            self.assertEqual({x['id'] for x in points if x['source_family']=='corpus'},{'0','1','2'})
            probes=[x for x in points if x['source_family']=='probe']
            self.assertEqual([x['activation'] for x in probes],[-4,3])
            self.assertTrue(all(x['cluster']=='unlabelled-probes' for x in probes))
            self.assertEqual(len(probes[0]['sources']),2)
            self.assertEqual(data,original)
            self.assertEqual(sum(sum(q.values()) for q in b['anatomy']['quadrants'].values()),5)

    def test_initial_interpretation_never_keeps_completed_experimental_points(self):
        from launch_files import initial_html
        from report import source_bundle
        b={'neuron':{'id':'L1_N2'},'distribution':{'n_tokens':10},
           'anatomy':{'points':[{'id':'1','gate':1,'up':2},{'id':'probe-1','source_family':'probe','gate':2,'up':3}],
                      'groups':[{'name':'Previous hypothesis'}],'quadrants':{'Previous hypothesis':{}},'selection':{'policy':'old'}}}
        source='<html><body><script id="bundle" type="application/json">'+json.dumps(b)+'</script><script>const B = JSON.parse(document.getElementById(\'bundle\').textContent);\nconst POS = "red";</script></body></html>'
        clean,_=source_bundle(initial_html(source))
        self.assertEqual([p['id'] for p in clean['anatomy']['points']],['1'])
        self.assertNotIn('groups',clean['anatomy'])
        self.assertNotIn('quadrants',clean['anatomy'])

if __name__=='__main__':unittest.main()
