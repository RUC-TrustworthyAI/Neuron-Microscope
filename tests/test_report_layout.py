"""Report layout regressions; saved fixtures only, no API or model calls."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from report_layout import headline,render,CIRCUIT,COLOURS,VERDICT_CODE,GRAPH_HELPERS

class LayoutTests(unittest.TestCase):
    def test_merge_relocates_bootstrap_after_earlier_layout_text_changes(self):
        from report import merge
        from unittest.mock import patch
        from html.parser import HTMLParser
        bundle={'neuron':{'id':'L1_N2'},'distribution':{'n_tokens':100},'claim':{'title':'Saved title'}}
        source='<html><head><style>old-style</style></head><body><main id="main"></main>'
        source+='<script type="application/json" id="bundle">'+json.dumps(bundle)+'</script>'
        source+="<script>\nconst B = JSON.parse(document.getElementById('bundle').textContent);\nconst POS = '#d9480f';\n</script></body></html>"
        class Scripts(HTMLParser):
            def __init__(self):super().__init__();self.active=False;self.contents=[]
            def handle_starttag(self,tag,attrs):
                if tag=='script':self.active=True;self.contents.append('')
            def handle_endtag(self,tag):
                if tag=='script':self.active=False
            def handle_data(self,data):
                if self.active:self.contents[-1]+=data
        for text in ['x','longer replacement style']:
            with self.subTest(style=text),patch('spectrum.render',side_effect=lambda page:page.replace('old-style',text)):
                rendered,result=merge(source,{})
            parser=Scripts();parser.feed(rendered)
            scripts=[s for s in parser.contents if 'const B =' in s]
            self.assertEqual(len(scripts),1)
            self.assertIn("const POS = '#d9480f'",scripts[0])
            self.assertEqual(result['claim'],bundle['claim'])

    def test_combined_peak_census_keeps_both_tails_and_inputs(self):
        import subprocess
        script=(Path(__file__).resolve().parents[1]/'tools/peak_census_display.js').read_text()
        script+="\nconst p={n:30,kinds:[{kind:'word',n:30}],tokens:[{token:' a',n:30}]},n={n:30,kinds:[{kind:'word',n:10},{kind:'digit',n:20}],tokens:[{token:' a',n:10},{token:'1',n:20}]}; const before=JSON.stringify([p,n]);const c=combinedPeakCensus(p,n);console.log(JSON.stringify({c,unchanged:before===JSON.stringify([p,n])}));"
        out=json.loads(subprocess.run(['node','-e',script],check=True,capture_output=True,text=True).stdout)
        self.assertTrue(out['unchanged'])
        self.assertEqual(out['c']['n'],60)
        self.assertEqual({x['kind']:x['n'] for x in out['c']['kinds']},{'word':40,'digit':20})
        self.assertEqual({x['token']:x['n'] for x in out['c']['tokens']},{' a':40,'1':20})

    def test_graph_axis_labels_keep_attributed_cue_missed_by_uniform_ticks(self):
        import subprocess
        code=GRAPH_HELPERS+"\nconsole.log(JSON.stringify([...graphLabelPositions(Array(577).fill('x'),[{ctx_idx:518,influence:65},{ctx_idx:0,influence:100}],576,1.8)]));"
        result=subprocess.run(['node','-e',code],capture_output=True,text=True,check=True)
        self.assertIn(518,json.loads(result.stdout))
        self.assertNotEqual(518 % 15,0)

    def test_example_preview_preserves_anchor_and_full_saved_input(self):
        import subprocess
        code="const argmaxAbs=a=>a.indexOf(Math.max(...a));\n"+(Path(__file__).resolve().parents[1]/'tools/exemplar_display.js').read_text()
        code+="\nconst e={peak_index:350,token_activations:Array.from({length:500},(_,i)=>({token:String(i),activation:i===350?90:0}))};const p=exemplarPreview(e);console.log(JSON.stringify([p.token_activations.length,p.peak_index,p.token_activations[0].token,p.token_activations[p.peak_index].activation,e.token_activations.length]));"
        result=subprocess.run(['node','-e',code],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),[33,32,'318',90,500])

    def test_criticism_uses_short_badges_and_preserves_legacy_decisions(self):
        import subprocess
        values=['survived','narrowed','rejected','Survived with a narrower mechanism: measured qualification.',
                'Partially supported and broadened: a qualified account.',
                'Directionally supported but not sufficient to reproduce the extremes.',
                'Partial support for output engagement.', 'Unknown decision']
        code=VERDICT_CODE+'\nconsole.log(JSON.stringify('+json.dumps(values)+'.map(verdictOf)));'
        result=subprocess.run(['node','-e',code],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),['survived','narrowed','rejected','narrowed','narrowed','narrowed','narrowed',''])
        html='// 8. Criticism: compact table, each row expands to the result and evidence.\nold();\n// 9. Appendix.\n'
        rendered=render(html)
        self.assertIn("P($('b',null,'Verdict. '),original)",rendered)
        self.assertEqual(render(rendered),rendered)

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
        self.assertNotIn('\nold\n',twice)

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
    def test_combined_headline_uses_only_scored_span_and_keeps_weak_controls(self):
        from pole_selection import comparison_ladder
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'artifacts/sweeps-probe';folder.mkdir(parents=True)
            rows=[{'condition':0,'status':'complete','label':'candidate','replacement':'a',
                   'text':'prefix a','scored_positions':[1,2],'selected':{'product':5},
                   'trace':[{'token':'prefix','product':1000},{'token':'a','product':5},{'token':'b','product':-7}]},
                  {'condition':1,'status':'complete','label':'control','replacement':'c',
                   'text':'prefix c','scored_positions':[1],'selected':{'product':.4},
                   'trace':[{'token':'prefix','product':2000},{'token':'c','product':.4}]}]
            (folder/'conditions.json').write_text(json.dumps(rows))
            figure={'id':'probe','kind':'sweep','template':'prefix {slot}',
                    'results':[{'activation':5},{'activation':.4}]}
            result=comparison_ladder({'custom':{'figures':[figure]}},tmp)
            self.assertEqual([x['activation'] for x in result['fillers']],[-7,.4])
            self.assertEqual(result['fillers'][0]['peak_index'],2)
            self.assertEqual(json.loads((folder/'conditions.json').read_text()),rows)

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
