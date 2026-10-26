"""CPU-only checks for shared section data, local derivatives and isolated test inputs."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from common import write,read,Runtime
from paper_sections import prepare,factorize,local_graph,labelled
from experiment_worker import run
import blind_tests

class SectionsTests(unittest.TestCase):
    def test_saved_selection_keeps_ids_and_native_traces(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            doc={'token_ids':[1,2,3,4],'tokens':['a','b','c','d'],
                 'gate':[1,2,3,4],'value':[2,3,4,5],'product':[2,6,12,20]}
            write(p/'report.json',{'points':{'10':{'document':'d','position':2,'values':[3,4,12]}},
                'documents':{'d':doc},'scatter':[10],
                'statistics':{'product':{'max':20,'min':2,'count':4}}})
            plan=prepare({'anatomy':{'sentence':{'case_id':'10','before':1,'after':1}},
                          'atlas':{'cases':['10'],'context_tokens':2}},p)
            self.assertEqual(plan['atlas']['cases'][0]['token_ids'],[1,2,3])
            self.assertEqual(plan['atlas']['cases'][0]['source_start'],0)
            self.assertEqual(plan['atlas']['cases'][0]['source_position'],2)
            self.assertEqual(plan['_anatomy']['strip']['act'],[6,12,20])
            self.assertEqual(plan['_anatomy']['strip']['anchor_index'],1)

    def test_factor_fit_excludes_heldout_features_and_reports_zero_support(self):
        def row(i,split,unit,weight):
            return {'id':i,'split':split,'text':i,'trace':[{'token':i,'product':weight,'position':0}],
                    'selected':{'position':0,'product':weight},
                    'upstream':[{'layer':0,'neuron':unit,'weight':weight}]}
        rows=[row('a','train',1,2),row('b','train',2,3),row('c','test',999,100)]
        result=factorize(rows,k=2)
        self.assertNotIn(999,[f[1] for f in result['routes_matrix']['pos']['features']])
        self.assertIsNone(result['assignments']['c'])
        self.assertEqual(sum(r['n_graphs'] for r in result['routes']),2)
        self.assertEqual(result,factorize(rows,k=2))

    def test_anatomy_only_worker_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            def load(_):raise AssertionError('A model was requested')
            result=run({'output':str(folder),'experiments':[{'tool':'paper_sections',
                'request':{'id':'a','output':str(folder/'a'),'plan':{'_anatomy':{'n':1}},'limits':{}}}]},runtime_loader=load)
            self.assertEqual(result['model_loads'],0)
            self.assertEqual(result['experiments'][0]['status'],'complete')
            self.assertEqual(read(folder/'a/sections.json'),{'anatomy':{'n':1}})

    def test_labels_cannot_replace_measured_numbers(self):
        original={'routes':[{'id':'p1','pole':1,'n_graphs':2,'graph_ids':['a','b'],
                    'share':1.,'upstream':[],'examples':[]}],
                  'clusters':[{'id':'semantic','graph_ids':['b'],'n_graphs':1}]}
        result=labelled(original,{'routes':{'p1':{'name':'A name','description':'A story','share':999}}})
        self.assertEqual(result['routes'][0]['share'],1.)
        self.assertEqual(result['clusters'],original['clusters'])
        self.assertNotIn('name',original['routes'][0])

    def test_local_gradient_blocks_intervening_mlp_paths(self):
        import torch
        from torch import nn
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.act_fn=nn.SiLU()
                self.gate_proj=nn.Linear(1,1,bias=False)
                self.up_proj=nn.Linear(1,1,bias=False)
                self.down_proj=nn.Linear(1,1,bias=False)
                with torch.no_grad():
                    self.gate_proj.weight.fill_(.2);self.up_proj.weight.fill_(.3);self.down_proj.weight.fill_(.4)
            def forward(self,x):return self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x))
        class Core(nn.Module):
            def __init__(self):
                super().__init__();self.embed_tokens=nn.Embedding(8,1)
                with torch.no_grad():self.embed_tokens.weight.fill_(1)
                self.layers=nn.ModuleList([nn.Module() for _ in range(4)])
                for layer in self.layers:layer.mlp=MLP()
            def forward(self,input_ids,use_cache=False):
                h=self.embed_tokens(input_ids)
                for layer in self.layers:h=h+layer.mlp(h)
                return SimpleNamespace(last_hidden_state=h)
        class Model(nn.Module):
            def __init__(self):
                super().__init__();self.model=Core();self.lm_head=nn.Linear(1,8,bias=False)
        class Tok:
            def decode(self,ids,**kwargs):return ''.join('a' for _ in ids)
        runtime=Runtime(Model(),Tok(),{'devices':[]},{'layer':2,'index':0})
        result=local_graph(runtime,{'token_ids':[1],'position':0},[{'layer':3,'neuron':0}],8)
        h=torch.tensor(1.);states=[];products=[]
        for _ in range(4):
            states.append(h)
            a=torch.nn.functional.silu(.2*h)*.3*h;products.append(a);h=h+.4*a
        def slope(h):
            # Frozen SiLU coefficient and the half rule distribute a/h in total.
            return torch.nn.functional.silu(.2*h)*.3*h/h
        expected=float(products[0]*.4*slope(states[2]))
        full=expected*float(1+.4*slope(states[1]))
        measured=next(u['weight'] for u in result['upstream'] if u['layer']==0)
        self.assertAlmostEqual(measured,expected,places=7)
        self.assertGreater(abs(measured-full),1e-6)
        self.assertAlmostEqual(result['downstream'][0]['weight'],float(products[2]*.4*slope(states[3])),places=7)
        self.assertEqual(runtime.forwards,2)

    def test_blind_writer_only_receives_frozen_explanation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            class Session:
                root=p
                config={'settings':__import__('settings').load()}
                config['settings']['blind']['candidates_per_pole']=1
                prior={'interpretation':{'claim':{'title':'L55_N11703 SECRET_LABEL',
                    'lede':'A place feature.','one_paragraph':'It responds to a place.',
                    'negative_pole':'Other places.'}}}
                jobs=[]
                def run_experiments(self,args):
                    out=p/'measurement';out.mkdir()
                    request={'id':'s','output':str(out),'plan':{'_corpus':{'max':9,'min':-4},'search':args['paper']['search']}}
                    raw={'search':[]}
                    for region in args['paper']['search']:
                        value=2 if region['pole']=='positive' else -1
                        raw['search'].append({'pole':region['pole'],'candidates':[{'status':'complete',
                            'text':region['candidates'][0],'trace':[{'token':'x','product':value}],
                            'selected':{'position':0,'product':value}}]})
                    write(out/'raw.json',raw);write(out/'sections.json',{})
                    self.jobs.append({'tool':'paper_sections','request':request})
                    return {}
            seen=[]
            def fake(messages,folder,route,**kwargs):
                seen.append(messages)
                return {'positive':['test positive'],'negative':['test negative']}
            answer=blind_tests.run(Session(),{'candidates_per_pole':1},llm=fake)
            payload=json.dumps(seen)
            self.assertNotIn('SECRET_LABEL',payload)
            self.assertNotIn('L55_N11703',payload)
            self.assertNotIn('corpus',payload)
            self.assertEqual(answer['tests']['blind_max']['achieved'],2)
            self.assertEqual(answer['tests']['blind_min']['target'],-4)
            self.assertEqual(len(seen),1)

if __name__=='__main__':unittest.main()
