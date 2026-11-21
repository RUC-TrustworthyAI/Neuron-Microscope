import unittest
from unittest.mock import patch
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from common import Runtime
from relp import rules,graph,direct,local_edges,HalfProduct

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

    @staticmethod
    def bf16_runtime():
        torch.manual_seed(42)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=50, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        )).to(dtype=torch.bfloat16).eval()
        return Runtime(model, Tok(), {'devices': [], 'threads': 1, 'activation_offload': 'cpu'},
                       {'layer': 2, 'index': 3})

    def test_bf16_checkpoint_preserves_atlas_and_both_graph_roots(self):
        runtime = self.bf16_runtime()
        ids, pos, layer, neuron = [1, 2, 3, 4, 5, 6], 5, 2, 3
        readers = [{'layer': 3, 'neuron': 4}, {'layer': 3, 'neuron': 7}]
        with torch.no_grad(), runtime.capture() as native:
            expected_logits = runtime.forward(ids, pos)
        expected_atlas = direct(runtime, ids, pos, layer, neuron, reader_units=readers, top_units=8)
        self.assertTrue(expected_atlas[0])
        self.assertEqual([(row['layer'], row['neuron']) for row in expected_atlas[1]], [(3, 4), (3, 7)])
        self.assertTrue(any(row['weight'] != 0. for row in expected_atlas[1]))
        expected_graphs = {
            token_id: graph(runtime, ids, pos, layer, neuron, token_id=token_id, budget=8, embeddings=3)
            for token_id in (None, 7)
        }

        projections = [block.mlp.gate_proj for block in runtime.model.model.layers]
        gate_calls = []
        hooks = [projection.register_forward_pre_hook(lambda *_: gate_calls.append(None))
                 for projection in projections]
        runtime.settings['activation_offload'] = 'checkpoint'
        try:
            first_forward = runtime.forwards
            actual_atlas = direct(runtime, ids, pos, layer, neuron, reader_units=readers, top_units=8)
            self.assertEqual(actual_atlas, expected_atlas)
            self.assertEqual(runtime.forwards - first_forward, 3)
            self.assertGreater(len(gate_calls), 3 * len(projections))
            for token_id, (expected, baseline_logits, coverage) in expected_graphs.items():
                with self.subTest(root_token_id=token_id):
                    gate_calls.clear()
                    actual, logits, actual_coverage = graph(
                        runtime, ids, pos, layer, neuron, token_id=token_id, budget=8, embeddings=3)
                    self.assertTrue(expected['links'])
                    self.assertEqual(actual, expected)
                    self.assertTrue(torch.equal(logits, baseline_logits))
                    self.assertTrue(torch.equal(logits, expected_logits))
                    self.assertEqual(actual['activation'], float(native['product'][pos]))
                    self.assertEqual(actual_coverage, coverage)
                    self.assertTrue(all(row['native_calls'] == 1 for row in actual_coverage))
                    self.assertGreater(len(gate_calls), 2 * len(projections))
        finally:
            for hook in hooks:
                hook.remove()
        with torch.no_grad():
            self.assertTrue(torch.equal(runtime.forward(ids, pos), expected_logits))

    def test_checkpoint_restores_model_after_a_failed_gradient_query(self):
        runtime = self.bf16_runtime()
        runtime.settings['activation_offload'] = 'checkpoint'
        blocks = list(runtime.model.model.layers)
        original_forwards = [block.forward for block in blocks]
        with patch('torch.autograd.backward', side_effect=RuntimeError('failed VJP')):
            with self.assertRaisesRegex(RuntimeError, 'failed VJP'):
                direct(runtime, [1, 2], 1, 2, 3)
        self.assertEqual([block.forward for block in blocks], original_forwards)
        self.assertFalse(runtime.model.model.embed_tokens._forward_hooks)
        self.assertTrue(all(not block.mlp.down_proj._forward_pre_hooks for block in blocks))

    def test_checkpoint_releases_upstream_leaf_gradients(self):
        runtime = self.bf16_runtime()
        ids, pos, layer, neuron = [1, 2, 3, 4, 5, 6], 5, 3, 3
        expected = direct(runtime, ids, pos, layer, neuron, top_units=24)
        self.assertTrue(expected[0])
        runtime.settings['activation_offload'] = 'checkpoint'

        with patch('torch.autograd.backward', wraps=torch.autograd.backward) as backward, \
             patch('torch.autograd.grad', side_effect=AssertionError('upstream gradients were returned together')):
            self.assertEqual(direct(runtime, ids, pos, layer, neuron, top_units=24), expected)

        backward.assert_called_once()
        leaves = backward.call_args.kwargs['inputs']
        self.assertEqual(len(leaves), layer)
        self.assertTrue(all(leaf.is_leaf and leaf.grad is None for leaf in leaves))
        self.assertTrue(all(not leaf._post_accumulate_grad_hooks for leaf in leaves))

    def test_checkpoint_local_edges_skip_missing_sources_and_zero_denominator(self):
        runtime = self.bf16_runtime()
        with torch.no_grad():
            runtime.model.model.layers[2].mlp.up_proj.weight.zero_()
        nodes = [
            {'node_id': 'source', 'layer': 0, 'ctx_idx': 0, 'feature': 2, 'influence': 1.},
            {'node_id': 'target', 'layer': 2, 'ctx_idx': 1, 'feature': 3, 'influence': 1.},
        ]
        baseline = local_edges(runtime, [1, 2], 1, nodes, max_edges=8)
        self.assertEqual(baseline, [])
        runtime.settings['activation_offload'] = 'checkpoint'
        self.assertEqual(local_edges(runtime, [1, 2], 1, nodes, max_edges=8), baseline)

if __name__=='__main__':unittest.main()
