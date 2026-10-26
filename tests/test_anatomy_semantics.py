import copy,json,tempfile,unittest
from pathlib import Path
from anatomy_semantics import annotate,apply,probe_key
from spectrum import anatomy_labels
from report_layout import anatomy_colours

class AnatomySemanticsTests(unittest.TestCase):
    def bundle(self):
        return {'spectrum':{'groups':[{'id':'main','name':'Main meaning','pole':'positive'},
                                    {'id':'neutral','name':'Related control','pole':'either'}]},
            'anatomy':{'selection':{},'points':[
                {'id':'1','source_family':'corpus','cluster':'main','gate':1,'up':2,'activation':2,'text':'saved prefix'},
                {'id':'probe-0001','source_family':'probe','cluster':'unlabelled-probes',
                 'token_ids':[10,20],'position':0,'gate':2,'up':-3,'activation':-6,
                 'text':'prefix','input_text':'prefix target','peak_token':'prefix'}]}}

    def test_labels_do_not_follow_sign_and_reuse_survives_probe_id_change(self):
        original=self.bundle();calls=[]
        def llm(messages,*args,**kwargs):
            packet=json.loads(messages[1]['content']);calls.append(packet)
            self.assertEqual(packet['experimental_points'][0]['following_prediction_target'],' target')
            return {'labels':[{'id':'probe-0001','group':'main','reason':'The same semantic meaning despite opposite sign.'}],
                    'clusters':[{'id':'related','name':'Related meanings','member_groups':['main','neutral']}]}
        with tempfile.TemporaryDirectory() as tmp:
            semantics=annotate(original,Path(tmp)/'first',llm=llm)
            changed=copy.deepcopy(original);changed['anatomy']['points'][1]['id']='probe-0099'
            repeated=annotate(changed,Path(tmp)/'second',previous=semantics,llm=llm)
            self.assertEqual(len(calls),1)
            before=[(p['gate'],p['up'],p['activation']) for p in changed['anatomy']['points']]
            apply(changed,repeated);anatomy_labels(changed)
            self.assertEqual([p['cluster'] for p in changed['anatomy']['points']],['related','related'])
            self.assertEqual([p['semantic_group'] for p in changed['anatomy']['points']],['main','main'])
            self.assertEqual(before,[(p['gate'],p['up'],p['activation']) for p in changed['anatomy']['points']])
            self.assertEqual(sum(sum(q.values()) for q in changed['anatomy']['quadrants'].values()),2)
            self.assertEqual(original,self.bundle())

    def test_reference_cluster_inherits_first_member_even_when_cluster_is_either(self):
        import subprocess
        source="""const SPEC_COLOUR={main:'orange',neutral:'grey'};
const B={clusters:[]};const calls=[];
const polePalette=items=>Object.fromEntries(items.map(g=>[g.id,g.pole==='positive'?'orange':g.pole==='negative'?'blue':'grey']));
const clusterColour=id=>'unknown'; const clusterName=id=>id;const has=x=>!!x;
B.anatomy={clusters:[{id:'c',name:'Combined',pole:'either',member_groups:['main','neutral']}],points:[]};
(() => { const A = B.anatomy; if (!has(A)) return; const figs = []; const pts = A.points || [];
calls.push(clusterColour('c'));})();
console.log(JSON.stringify(calls));"""
        code=anatomy_colours(source)
        result=subprocess.run(['node','-e',code],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),['orange'])
        self.assertEqual(anatomy_colours(code),code)

if __name__=='__main__':unittest.main()
