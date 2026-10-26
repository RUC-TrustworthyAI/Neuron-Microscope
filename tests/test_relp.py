import unittest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from common import Runtime
from relp import rules,graph,HalfProduct

class Tok:
    def decode(self,ids):return str(ids)

class RelPTest(unittest.TestCase):
    def test_native_forward_and_multihop(self):
        torch.manual_seed(42)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=50,hidden_size=32,
            intermediate_size=48,num_hidden_layers=3,num_attention_heads=4,
            num_key_value_heads=2,head_dim=8)).eval()
        runtime=Runtime(model,Tok(),{'devices':[],'threads':1},{'layer':2,'index':3})
        ids=[1,2,3,4]
        with torch.no_grad():expected=runtime.forward(ids,3)
        with rules(model):actual=runtime.forward(ids,3)
        self.assertTrue(torch.equal(expected,actual))
        result,_,_=graph(runtime,ids,3,2,3,budget=20,embeddings=4)
        self.assertGreater(len({e['target'] for e in result['links']}),1)
        self.assertTrue(all(e['source']!=e['target'] for e in result['links']))
        with torch.no_grad():restored=runtime.forward(ids,3)
        self.assertTrue(torch.equal(expected,restored))
        self.assertEqual(model.config._attn_implementation,'sdpa')

    def test_half_rule(self):
        x=torch.tensor(2.,requires_grad=True);y=torch.tensor(-3.,requires_grad=True)
        z=HalfProduct.apply(x,y)
        dx,dy=torch.autograd.grad(z,[x,y])
        self.assertEqual(float(x*dx+y*dy),float(z))

if __name__=='__main__':unittest.main()
