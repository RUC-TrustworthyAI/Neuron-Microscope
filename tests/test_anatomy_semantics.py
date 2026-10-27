import copy,json,tempfile,unittest
from pathlib import Path
from anatomy_semantics import annotate,apply,probe_key
from spectrum import anatomy_labels
from report_layout import anatomy_colours

class AnatomySemanticsTests(unittest.TestCase):
    def test_model_authored_merge_reassigns_existing_labels_without_losing_points(self):
        bundle=self.bundle();prior={'groups':[{'id':'one-document','name':'One document topic','pole':'either'}],
            'assignments':{probe_key(bundle['anatomy']['points'][1]):{'id':'old','group':'one-document'}},
            'clusters':[]}
        def llm(*args,**kwargs):
            return {'group_merges':[{'source':'one-document','target':'neutral','reason':'Same background pattern.'}],
                    'labels':[], 'clusters':[{'id':'main-cluster','name':'Main','member_groups':['main']},
                                            {'id':'background','name':'Background','member_groups':['neutral']}]}
        before=copy.deepcopy(bundle)
        with tempfile.TemporaryDirectory() as tmp:
            result=annotate(bundle,Path(tmp),previous=prior,llm=llm)
            self.assertEqual(result['aliases'],{'one-document':'neutral'})
            self.assertEqual(result['assignments'][probe_key(bundle['anatomy']['points'][1])]['group'],'neutral')
            self.assertNotIn('one-document',{g['id'] for g in result['groups']})
            apply(bundle,result)
        self.assertEqual(len(bundle['anatomy']['points']),len(before['anatomy']['points']))
        self.assertEqual([(p['gate'],p['up'],p['activation']) for p in bundle['anatomy']['points']],
                         [(p['gate'],p['up'],p['activation']) for p in before['anatomy']['points']])
        self.assertEqual(bundle['anatomy']['points'][1]['cluster'],'background')

    def test_unassigned_groups_share_one_unresolved_display_cluster(self):
        bundle=self.bundle();bundle['spectrum']['groups'] += [
            {'id':'extra-'+str(i),'name':'Unresolved '+str(i),'pole':'either'} for i in range(20)]
        def llm(*args,**kwargs):
            return {'labels':[{'id':'probe-0001','group':'extra-0'}],
                    'clusters':[{'id':'main-cluster','name':'Main','member_groups':['main']}]}
        with tempfile.TemporaryDirectory() as tmp:
            result=annotate(bundle,Path(tmp),llm=llm);apply(bundle,result)
        self.assertEqual(len(result['groups']),22)
        self.assertEqual(len(result['clusters']),2)
        self.assertEqual(len(result['clusters'][-1]['unassigned_groups']),21)
        self.assertEqual(bundle['anatomy']['points'][1]['semantic_group'],'extra-0')
        self.assertEqual(bundle['anatomy']['points'][1]['cluster'],'unresolved-cluster-assignment')

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

    def test_colours_follow_predictions_when_measurements_disagree(self):
        import subprocess
        source="""const POS='orange',NEG='blue';
const semanticPalette = () => ({});
const B={clusters:[]};const calls=[];
const clusterColour=id=>'unknown'; const clusterName=id=>id;const has=x=>!!x;
B.anatomy={clusters:[{id:'c',name:'Combined',pole:'negative',member_groups:['main']}],points:[{cluster:'c',activation:300,predicted_pole:'negative'},{cluster:'c',activation:-50,predicted_pole:'positive'},{cluster:'c',activation:-25}]};
(() => { const A = B.anatomy; if (!has(A)) return; const figs = []; const pts = A.points || [];
calls.push(...pts.map(p=>({fill: clusterColour(p.cluster)})),clusterColour('c'));})();
console.log(JSON.stringify(calls));"""
        code=anatomy_colours(source)
        result=subprocess.run(['node','-e',code],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),[{'fill':'blue'},{'fill':'orange'},{'fill':'#8a8a90'},'linear-gradient(90deg,orange 0 50%,blue 50% 100%)'])
        self.assertEqual(anatomy_colours(code),code)

if __name__=='__main__':unittest.main()
