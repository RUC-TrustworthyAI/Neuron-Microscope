import json
import tempfile
import unittest
from pathlib import Path
from report_completion import corpus_counts, minimal_pairs, unit_labels, full_reading, measured_contexts
from launch_files import initial_html
from report import source_bundle

class ReportCompletionTests(unittest.TestCase):
    def test_display_context_reuses_exact_measured_trace_and_steering_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'artifacts/graphs-g1';folder.mkdir(parents=True)
            trace=[{'token':'old','product':0.}, {'token':' cue','product':12.}]
            result={'status':'complete','trace':trace,'graph':{'position':1},
                    'steering':{'prompt':'old cue'}}
            (folder/'result.json').write_text(json.dumps(result))
            b={'graphs_curated':{'picks':[{'id':'g1','graph':{'position':1}}]},
               'tests':{'steering':[{'prompt':'old cue','baseline':.2,'steered':.4,'strength':2}]}}
            measured_contexts(b,root)
            row=b['tests']['steering'][0]
            self.assertEqual((row['baseline'],row['steered'],row['strength']),(.2,.4,2))
            self.assertEqual(row['activation_context']['activation'],12.)
            self.assertEqual(row['activation_context']['peak_index'],1)
            self.assertEqual(b['graphs_curated']['picks'][0]['graph']['token_activations'],
                             [{'token':'old','activation':0.},{'token':' cue','activation':12.}])

    def test_corpus_mass_is_separate_from_selected_semantic_samples(self):
        bundle={'bigscan':{'hist_centers':[-2,0,3],'hist':[2,100,3]},
                'spectrum':{'edges':[-4,-1,1,4],'sample_hist':{'counts':[10,1,10]}}}
        corpus_counts(bundle)
        self.assertEqual(bundle['spectrum']['corpus_hist']['counts'],[2,100,3])
        self.assertEqual(bundle['spectrum']['sample_hist']['counts'],[10,1,10])
        self.assertEqual(bundle['spectrum']['corpus_hist']['n_tokens'],105)

    def test_pairs_reuse_saved_traces_in_one_frame(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            (folder/'figure.json').write_text(json.dumps({'id':'s1','template':'A {slot} B'}))
            rows=[{'status':'complete','condition':i,'replacement':word,'text':'A '+word+' B',
                   'selected':{'product':score},'trace':[{'token':word,'gate':i,'value':2,'product':score}]}
                  for i,(word,score) in enumerate([('x',4),('y',-2),('z',1)])]
            (folder/'conditions.json').write_text(json.dumps(rows))
            b={'anatomy':{}}
            minimal_pairs(b,[folder])
            self.assertEqual([x['condition'] for x in b['anatomy']['headline']],[0,1])
            self.assertEqual(b['anatomy']['headline'][0]['act'],[4])

    def test_explicit_agent_label_is_preserved_and_route_role_is_named(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'route-factors.json').write_text(json.dumps({'assignments':{'a':'p1'}}))
            (p/'atlas-raw.json').write_text(json.dumps([{'id':'a','status':'complete','upstream':[{'layer':0,'neuron':2,'weight':4}], 'downstream':[]}]))
            b={'routes':[{'id':'p1','name':'Named route'}],
               'circuit':{'upstream':[{'layer':0,'neuron':2},{'layer':0,'neuron':3,'label':'Existing agent label'}]}}
            unit_labels(b,[p])
            self.assertEqual(b['circuit']['upstream'][0]['label'],'Writer in Named route')
            self.assertEqual(b['circuit']['upstream'][1]['label'],'Existing agent label')

    def test_expanded_reader_reuses_exact_existing_text(self):
        original={'one_line':'Existing assessment','poles':{'positive':{'group':'unresolved','notes':'Existing unresolved note'}}}
        b={'agent_read':original}
        full_reading(b)
        self.assertEqual(b['agent_read_full']['main_job'],original['one_line'])
        self.assertEqual(b['agent_read_full']['remainders'][0]['evidence'],'Existing unresolved note')
        self.assertNotIn('main_job',original)

class IndependentInputTests(unittest.TestCase):
    def test_initial_html_drops_labels_and_both_reader_views(self):
        bundle={'neuron':{'id':'L1_N2','layer':1,'neuron':2,'autointerp_label':'hidden label','autointerp_description':'hidden description'},
                'distribution':{'n_tokens':100},'agent_read':{'one_line':'old read'},
                'agent_read_full':{'main_job':'old expanded read'},
                'anatomy':{'headline':[],'headline_source':{'template':'old experiment'}}}
        html='<html><body><script id="bundle" type="application/json">'+json.dumps(bundle)+'</script><script>const B = JSON.parse(document.getElementById(\'bundle\').textContent);\nconst POS = "red";</script></body></html>'
        page=initial_html(html)
        cleaned,_=source_bundle(page)
        self.assertNotIn('autointerp_label',cleaned['neuron'])
        self.assertNotIn('autointerp_description',cleaned['neuron'])
        self.assertIsNone(cleaned.get('agent_read'))
        self.assertIsNone(cleaned.get('agent_read_full'))
        self.assertNotIn('headline_source',cleaned['anatomy'])
        self.assertNotIn('hidden label',page)
        self.assertNotIn('old expanded read',page)

if __name__=='__main__':unittest.main()
