"""Spectrum context starts 128 tokens before visible activation or its anchor."""
import json
import subprocess
import unittest
from exemplar_display import HELPER, render


class SpectrumDisplayTests(unittest.TestCase):
    def test_first_highlight_and_weak_anchor_keep_full_saved_data(self):
        script="""const make = (peak, values) => ({peak_index:peak,activation:values[peak] || 0,
  cut_right:true,token_activations:Array.from({length:400},(_,i)=>({token:String(i)+' ',activation:values[i] || 0}))});
const samples=[make(350,{200:20,350:90}),make(350,{350:.23}),make(350,{200:-20,350:-90}),make(10,{10:90})];
const original=JSON.stringify(samples);
const rows=samples.map(e=>exemplarDisplay(e,100,128)).map(e=>({start:e.token_activations[0].token,
  length:e.token_activations.length,peak:e.peak_index,anchor:e.token_activations[e.peak_index].activation,
  left:!!e.cut_left,right:e.cut_right}));
console.log(JSON.stringify({rows,unchanged:original===JSON.stringify(samples)}));
"""
        # Call before the declaration, as the spectrum renders before Exemplars.
        result=subprocess.run(['node','-e',script+HELPER.read_text()],check=True,capture_output=True,text=True)
        out=json.loads(result.stdout)
        self.assertTrue(out['unchanged'])
        self.assertEqual([r['start'] for r in out['rows']],['72 ','222 ','72 ','0 '])
        self.assertEqual([r['peak'] for r in out['rows']],[278,128,278,10])
        self.assertEqual([r['anchor'] for r in out['rows']],[90,.23,-90,90])
        self.assertEqual([r['left'] for r in out['rows']],[True,True,True,False])
        self.assertTrue(all(r['right'] for r in out['rows']))

    def test_renderer_applies_window_only_to_spectrum_and_is_idempotent(self):
        row='exRow(e, s, (e.activation || 0) < 0, tileScale)'
        source='<html><head></head><body>\n// 1b. Activation spectrum\n'+row
        source+='\n// 2. Exemplars\nunchanged();\n// 3. Search\n'+row+'</body></html>'
        once=render(source)
        self.assertIn('exRow(exemplarDisplay(e, s, 128), s, (e.activation || 0) < 0, tileScale)',once)
        self.assertEqual(once.count(row),1)
        self.assertEqual(render(once),once)
        self.assertEqual(render(once.replace('exemplarDisplay(e, s, 128)','exemplarDisplay(e, s, 32)')),once)


if __name__=='__main__':unittest.main()
