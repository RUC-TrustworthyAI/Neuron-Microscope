"""Report layout regressions; saved fixtures only, no API or model calls."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from report_layout import headline,render,CIRCUIT,COLOURS

class LayoutTests(unittest.TestCase):
    def test_circuit_uses_two_independent_slots(self):
        self.assertIn("table(G.upstream,",CIRCUIT)
        self.assertIn("table(G.downstream,",CIRCUIT)
        self.assertNotIn("G.logits",CIRCUIT)
        self.assertIn("if (!has(items)) return $('span', {class:'tile'}, '—');",CIRCUIT)

    def test_repeated_stage_render_does_not_duplicate_test_slots(self):
        source="// 6. Circuit tables: old\nold\n// 7. Tests.\n"
        source+="const T = B.tests; if (!has(T)) return; const figs = [];\n  if (has(T.steering))"
        once=render(source);twice=render(once)
        self.assertEqual(twice.count("missing('Route prediction')"),1)
        self.assertEqual(twice.count('id="paper-layout-order"'),1)
        self.assertNotIn('old',twice)

    def test_publication_removes_extra_scan_without_removing_distribution(self):
        source="// 1. Distribution\nmain_distribution();\n"
        source+="  // Larger scan: secondary pass\nconst big = []; render_extra_scan();\n"
        source+="  sec({ id: 'distribution', wide: true }, 'Activation distribution', why, fig(svg), tiles, big); })();"
        rendered=render(source)
        self.assertIn("main_distribution();",rendered)
        self.assertIn("'Activation distribution'",rendered)
        self.assertNotIn("render_extra_scan",rendered)
        self.assertNotIn("Larger scan",rendered)
        self.assertNotIn("tiles, big",rendered)

    def test_colours_use_only_explicit_poles(self):
        import subprocess
        script = "const B={spectrum:{points:[{group:'a',activation:-100},{group:'b',activation:100}]}};\n"
        script += COLOURS
        script += "\nconsole.log(JSON.stringify([poleOf({id:'a',pole:'positive'}),poleOf({id:'b',pole:'negative'}),poleOf({id:'b',pole:'either'}),poleOf({id:'b'}),isOther({id:'other'}),isOther({id:'other_states'})]));"
        result = subprocess.run(['node','-e',script],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),['positive','negative','either','either',True,False])
        self.assertNotIn('median',COLOURS)
        self.assertNotIn('DISPLAY_GROUP_VALUES',COLOURS)

    def test_old_median_adapter_is_replaced_and_rendering_is_idempotent(self):
        source = "// Shared display colours. Missing pole metadata uses the saved group median.\n"
        source += "const DISPLAY_GROUP_VALUES = {}; old_median_fallback();\n"
        source += "// Pole-and-order palette shared by the spectrum and the clusters.\n"
        once=render(source); twice=render(once)
        self.assertNotIn('DISPLAY_GROUP_VALUES',twice)
        self.assertNotIn('old_median_fallback',twice)
        self.assertEqual(twice.count('const poleOf ='),1)
        self.assertEqual(once,twice)

    def test_absent_routes_have_no_extra_tiles_but_real_factorization_stays(self):
        source="clusters:['Clusters',['Clusters','Routes','Route factorization']]\n"
        source+="if(raw.length) summary('Raw factorization');"
        result=render(source)
        self.assertIn("clusters:['Clusters',['Clusters']]",result)
        self.assertNotIn("'Routes','Route factorization'",result)
        self.assertIn("summary('Raw factorization')",result)

    def test_header_reuses_saved_values_and_updates_caption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'artifacts/sweeps-probe';p.mkdir(parents=True)
            (p/'result.json').write_text(json.dumps({'conditions':[{'condition':0,'status':'complete',
                'text':'same word word','trace':[{'token':'same','product':0.},
                 {'token':' word','product':3.},{'token':' word','product':1.}]}]}))
            figure={'id':'probe','kind':'sweep','template':'same {slot}','caption':'Measured caption',
                'results':[{'tag':'word','text':'same word word','activation':3.,'peak_index':1,'peak_token':' word'}]}
            bundle={'headline':{'strip':{'tokens':['saved']}},'custom':{'figures':[figure]}}
            h=headline(bundle,tmp)
            self.assertEqual(h['ladder']['fillers'][0]['activation'],3.)
            self.assertEqual(h['ladder']['fillers'][0]['peak_index'],1)
            self.assertEqual(h['ladder']['fillers'][0]['values'],[0.,3.,1.])
            bundle['headline']=h;figure['caption']='Revised caption'
            self.assertEqual(headline(bundle,tmp)['ladder']['caption'],'Revised caption')
            self.assertEqual(h['strip'],{'tokens':['saved']})

class SignedSelectionTests(unittest.TestCase):
    def test_contrast_and_intent_prevent_incidental_negative_control_selection(self):
        from pole_selection import select
        b={'distribution':{'corpus_min':-60,'corpus_max':400},'custom':{'figures':[
            {'id':'flat','kind':'sweep','pole':'positive','results':[{'activation':200},{'activation':195}]},
            {'id':'positive','kind':'sweep','pole':'positive','results':[{'activation':100},{'activation':-1}]},
            {'id':'negative','kind':'sweep','pole':'negative','results':[{'activation':-50},{'activation':-.2}]}]}}
        picks=select(b,'/nonexistent')
        self.assertEqual(picks['positive']['figure']['id'],'positive')
        self.assertEqual(picks['negative']['figure']['id'],'negative')

    def test_headline_selects_both_saved_signs_and_negative_anchor(self):
        data={'points':{'p':{'document':'1','position':0,'values':[1,4,4]},
                        'n':{'document':'2','position':0,'values':[2,-3,-6]}},
              'documents':{'1':{'tokens':['pos'],'product':[4]},'2':{'tokens':['neg'],'product':[-6]}},
              'examples':{'top_documents':['p'],'bottom_documents':['n']}}
        b={'headline':{'ladder':{'source_figure_id':'old'}},'custom':{'figures':[
            {'id':'p','kind':'sweep','pole':'positive','results':[{'activation':4},{'activation':1}]},
            {'id':'n','kind':'sweep','pole':'negative','results':[{'activation':-6},{'activation':0}]}]}}
        h=headline(b,'/nonexistent',data=data)
        self.assertEqual([p['pole'] for p in h['poles']],['positive','negative'])
        self.assertEqual([p['strip']['case_id'] for p in h['poles']],['p','n'])
        self.assertEqual(h['poles'][1]['ladder']['source_figure_id'],'n')

    def test_only_positive_corpus_has_no_invented_negative_panel(self):
        from pole_selection import select
        b={'distribution':{'corpus_min':0,'corpus_max':4},'custom':{'figures':[
            {'id':'n','kind':'sweep','pole':'negative','results':[{'activation':1},{'activation':0}]}]}}
        self.assertNotIn('negative',select(b,'/nonexistent'))

    def test_new_measured_sign_is_kept_even_outside_corpus_range(self):
        from pole_selection import select
        b={'distribution':{'corpus_min':0,'corpus_max':4},'custom':{'figures':[
            {'id':'new','kind':'sweep','pole':'negative','results':[{'activation':-3},{'activation':1}]}]}}
        self.assertEqual(select(b,'/nonexistent')['negative']['figure']['id'],'new')

if __name__=='__main__':unittest.main()
