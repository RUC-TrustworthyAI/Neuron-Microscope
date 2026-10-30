import copy,json,tempfile,unittest
from pathlib import Path
from anatomy_selection import build, SHARED_POLICY
from spectrum import anatomy_labels

class AnatomySelectionTests(unittest.TestCase):
    def test_views_keep_the_same_shared_points_across_measurements(self):
        from blind_anatomy import apply,point_identity
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scan=root/'scan';scan.mkdir()
            data={'points':{'0':{'document':'d','position':0,'values':[2,3,6]},
                            '1':{'document':'d','position':1,'values':[3,-2,-6]}},
                  'documents':{'d':{'tokens':['first',' second']}},
                  'examples':{'top_tokens':['0'],'bottom_tokens':['1']},'bands':[]}
            def sample(i):
                return {'id':'spectrum-'+str(i),'source_family':'corpus',
                    'document':'d' if i==0 else str(i),'position':0,'gate':2,'up':3,
                    'activation':6,'group':'g','bin':1,'peak_token':'first',
                    'observed_prefix':'Complete prefix '+str(i)}
            shared=[sample(i) for i in range(500)]
            bundle={'spectrum':{'points':shared,'groups':[{'id':'g','name':'Context'}],
                     'bins':[{'n_labeled':0,'groups':{}},{'n_labeled':500,'groups':{'g':500}}]}}
            feedback={**sample(900),'id':'old-feedback','source':'corpus:scan','cluster':'unlabelled',
                      'evaluation_split':'feedback'}
            heldout={**sample(901),'id':'heldout','source':'corpus:scan','cluster':'unlabelled',
                     'evaluation_split':'final_holdout'}
            saved={'points':[feedback],'edges':[-10,0,10],'seed':42,'scan_seconds':0,'tokens_scanned':1000}
            path=scan/'anatomy-scan.json';path.write_text(json.dumps(saved))
            request={'report':str(root),'sections':[str(scan)]}
            build(bundle,data,request)
            points=bundle['anatomy']['points']
            self.assertEqual(len(points),500)
            self.assertEqual(sum(p['source']=='corpus:spectrum' for p in points),500)
            self.assertNotIn('0',{p['id'] for p in points})
            self.assertNotIn('1',{p['id'] for p in points})
            self.assertNotIn('old-feedback',{p['id'] for p in points})
            self.assertEqual(bundle['anatomy']['selection']['policy'],SHARED_POLICY)
            records=[{'id':p['id'],'identity':point_identity(p),'group':'g','predicted_pole':'negative',
                      'reason':'Frozen rule prediction.'} for p in points]
            apply(bundle,{'version':'v','response_rules':[{'id':'g','name':'Context','predicted_pole':'negative'}],
                          'records':records,'summary':{}})
            byid={p['id']:p for p in bundle['anatomy']['points']}
            for p in shared:
                q=byid[p['id']]
                self.assertEqual((q['gate'],q['up'],q['activation']),(p['gate'],p['up'],p['activation']))
                self.assertEqual((q['cluster'],q['predicted_pole'],q['prediction_reason']),
                                 (p['group'],p['predicted_pole'],p['prediction_reason']))
            saved['points'].append(heldout);path.write_text(json.dumps(saved))
            build(bundle,data,request)
            self.assertEqual(len(bundle['anatomy']['points']),500)
            self.assertNotIn('heldout',{p['id'] for p in bundle['anatomy']['points']})
            measured={'status':'complete','text':'probe','token_ids':[10],
                'selected':{'position':0,'gate':2,'value':3,'product':6,'token':'probe','offset':[0,5]}}
            (scan/'search-raw.json').write_text(json.dumps([{'id':'r','candidates':[measured],'controls':[]}]))
            build(bundle,data,request)
            self.assertEqual(len(bundle['anatomy']['points']),500)
            self.assertEqual(sum(p.get('source_family')=='probe' for p in bundle['anatomy']['points']),0)

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
